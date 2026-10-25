/* The five radiation monitors: descriptive details shared by the 3D model (reactor3d.js) and the
   exported report (report_data.js). Placement in the 3D scene stays in reactor3d.js.
   typical = normal dose rate from the ideal healthy run (µSv/h). Runs in the browser
   (window.NuclearRadSensors) and in Node (module.exports). */
(function (root) {
  "use strict";

  const SENSORS = [
    { id: "RAD-01", name: "Reactor cavity area monitor", type: "Gamma area monitor · ionisation chamber",
      where: "Operating deck, beside the reactor head", typical: 2.5 },
    { id: "RAD-02", name: "Primary loop area monitor", type: "Gamma area monitor · Geiger-Müller tube",
      where: "Steam generator compartment, next to a reactor coolant pump", typical: 8.0 },
    { id: "RAD-03", name: "Personnel airlock monitor", type: "Gamma area monitor · plastic scintillator",
      where: "Inside the personnel airlock entrance", typical: 0.8 },
    { id: "RAD-04", name: "Plant vent stack monitor", type: "Effluent monitor · NaI(Tl) scintillation detector",
      where: "Vent stack sampling platform, outside containment", typical: 0.25 },
    { id: "RAD-05", name: "Containment high-range monitor", type: "High-range gamma monitor · ionisation chamber",
      where: "Upper containment wall, below the dome", typical: 1.5 },
  ];

  if (typeof module !== "undefined" && module.exports) module.exports = SENSORS;
  else root.NuclearRadSensors = SENSORS;
})(typeof window !== "undefined" ? window : globalThis);
