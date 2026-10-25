/* Trust levels and recommended actions for the exported report.

   Everything here is deterministic: the same analysis always gives the same trust levels and the
   same actions, in the same order. No model calls, no network. To change what the report
   recommends, edit TRUST_RULES / ACTION_RULES / the limits in SETTINGS below; each rule says what
   evidence triggers it and what it recommends.

   Input: the per-sensor rows built by report_data.js (one object per sensor) and the plant-level
   verdict from detector.js. Runs in the browser (window.ReportRules) and in Node (module.exports). */
(function (root) {
  "use strict";

  const SETTINGS = {
    OFFLINE_TAIL_FRACTION: 0.05,  // offline = the last 5% of readings (at least OFFLINE_MIN_READINGS) are all missing
    OFFLINE_MIN_READINGS: 5,
    MONITOR_HEALTH: 80,           // a healthy sensor whose health score is below this gets a "monitor" note
    DATA_GAP_SHARE: 0.05,         // more than 5% missing readings over the file is worth a data-logging check
  };

  // ------------------------------------------------------------------ trust levels
  // Ordered from best to worst. A sensor gets the first level whose test passes, checking from the
  // worst level down (Offline before Untrusted before Needs review).
  const TRUST_LEVELS = ["Trusted", "Needs review", "Untrusted", "Offline"];
  const TRUST_RULES = [
    { level: "Offline", test: (s) => s.offline,
      why: (s) => `No reading for the last ${s.silentReadings} readings` },
    { level: "Untrusted", test: (s) => s.status === "Faulty",
      why: (s) => `Currently showing ${String(s.fault || "a fault").toLowerCase()}` },
    { level: "Needs review", test: (s) => s.status === "Watch" || s.status === "At risk",
      why: (s) => (s.status === "At risk" ? `Trending towards ${String(s.fault).toLowerCase()}` : `Intermittent ${String(s.fault || "issue").toLowerCase()} earlier in the data`) },
    { level: "Trusted", test: () => true, why: () => "Agrees with its peers" },
  ];

  function trustLevel(sensor) {
    const rule = TRUST_RULES.find((r) => r.test(sensor));
    return { level: rule.level, reason: rule.why(sensor) };
  }

  // ------------------------------------------------------------------ recommended actions
  // Each rule looks at one sensor (scope "sensor") or at the whole plant (scope "plant") and returns
  // the evidence string when it applies, or null. Rules in the same `group` are exclusive for a
  // sensor: only the first matching one in this list is used (e.g. a sensor marked for replacement
  // does not also get a recalibration action).
  const PRIORITY_ORDER = { High: 0, Medium: 1, Low: 2 };
  const ACTION_RULES = [
    {
      id: "plant-event", scope: "plant", priority: () => "High", title: "Escalate a plant-wide event",
      when: (plant) => (plant.verdict && plant.verdict.level === "event" ? plant.verdict.text : null),
      next: () => "Escalate to the operating team through approved procedures and confirm with independent instruments. Do not treat the shared change as a sensor fault.",
    },
    {
      id: "offline", scope: "sensor", group: "maintenance", priority: () => "High", title: "Investigate a silent sensor",
      when: (s) => (s.offline ? `No reading for the last ${s.silentReadings} readings (last seen at ${s.lastSeenLabel ?? "—"}).` : null),
      next: () => "Check the sensor's power supply, cable, connector and the data logger channel; restore or replace the channel.",
    },
    {
      id: "replace", scope: "sensor", group: "maintenance", priority: () => "High", title: "Replace the sensor",
      when: (s) => (s.maintenance && s.maintenance.action === "Replace" ? s.maintenance.text : null),
      next: (s) => (s.fault === "Stuck" ? "Replace the sensor, then confirm the new one tracks its peers." : "Replace the detector, recalibrate the replacement and confirm it agrees with its peers."),
    },
    {
      id: "repair", scope: "sensor", group: "maintenance",
      priority: (s) => (s.maintenance.urgency === "Immediate" ? "High" : "Medium"), title: "Repair the sensor or its connection",
      when: (s) => (s.maintenance && s.maintenance.action === "Repair" ? s.maintenance.text : null),
      next: (s) => (s.fault === "Noisy" ? "Repair cable shielding and grounding and check the supply for ripple; replace if the noise persists."
        : s.fault === "No signal" ? "Repair the cable, connector, power or logger channel; replace the sensor if the signal does not return."
        : "Inspect and repair connectors and cabling; replace the sensor if the problem persists."),
    },
    {
      id: "recalibrate", scope: "sensor", group: "maintenance", priority: () => "Medium", title: "Recalibrate the sensor",
      when: (s) => (s.maintenance && s.maintenance.action === "Recalibrate" ? s.maintenance.text : null),
      next: () => "Recalibrate against a check source or reference instrument; plan a replacement if the drift returns.",
    },
    {
      id: "at-risk", scope: "sensor", group: "maintenance", priority: () => "Medium", title: "Recalibrate before the drift limit",
      when: (s) => (s.status === "At risk" ? s.summary : null),
      next: () => "Schedule a recalibration before the projected crossing and re-check the trend afterwards.",
    },
    {
      id: "inspect", scope: "sensor", group: "maintenance", priority: () => "Low", title: "Inspect at the next routine check",
      when: (s) => (s.status === "Watch" ? s.summary : null),
      next: (s) => (s.fault === "Spike" ? "Inspect connectors, cabling and grounding for intermittent contacts or interference."
        : s.fault === "No signal" ? "Inspect the connection and review data-logger errors around the gaps."
        : "Inspect the sensor and compare it against an independent reading."),
    },
    {
      id: "monitor", scope: "sensor", group: "maintenance", priority: () => "Low", title: "Keep monitoring",
      when: (s) => (s.trust === "Trusted" && s.health !== null && s.health < SETTINGS.MONITOR_HEALTH
        ? `Health score ${s.health}/100 is below ${SETTINGS.MONITOR_HEALTH} although the sensor agrees with its peers.` : null),
      next: () => "No maintenance needed now; review the trend in the next report.",
    },
    {
      id: "data-gaps", scope: "sensor", group: "data", priority: () => "Low", title: "Check data logging",
      when: (s) => (!s.offline && s.missingShare > SETTINGS.DATA_GAP_SHARE
        ? `${(100 * s.missingShare).toFixed(1)}% of this sensor's readings are missing.` : null),
      next: () => "Review the logger and network for dropped samples on this channel.",
    },
  ];

  /** Recommended actions, sorted High → Medium → Low, then by sensor ID. */
  function recommendActions(rows, plant = {}) {
    const actions = [];
    for (const rule of ACTION_RULES.filter((r) => r.scope === "plant")) {
      const evidence = rule.when(plant);
      if (evidence) actions.push({ rule: rule.id, priority: rule.priority(plant), title: rule.title,
        sensors: rows.map((r) => r.id), reason: evidence, nextStep: rule.next(plant) });
    }
    for (const row of rows) {
      const usedGroups = new Set();
      for (const rule of ACTION_RULES.filter((r) => r.scope === "sensor")) {
        if (rule.group && usedGroups.has(rule.group)) continue;
        const evidence = rule.when(row);
        if (!evidence) continue;
        if (rule.group) usedGroups.add(rule.group);
        actions.push({ rule: rule.id, priority: rule.priority(row), title: rule.title, sensors: [row.id],
          reason: evidence, nextStep: rule.next(row) });
      }
    }
    return actions.sort((a, b) => PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority]
      || (a.sensors.length > 1 ? -1 : 0) - (b.sensors.length > 1 ? -1 : 0)
      || String(a.sensors[0]).localeCompare(String(b.sensors[0])));
  }

  const api = { SETTINGS, TRUST_LEVELS, TRUST_RULES, ACTION_RULES, PRIORITY_ORDER, trustLevel, recommendActions };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ReportRules = api;
})(typeof window !== "undefined" ? window : globalThis);
