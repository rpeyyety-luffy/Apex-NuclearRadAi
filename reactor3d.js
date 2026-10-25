// 3D model of a pressurised-water reactor containment with the five radiation monitors.
// The analysis in app.js reports results through the "core-results" event; each monitor's
// beacon, screen and label then show its status. Placement is illustrative.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

// ------------------------------------------------------------------ the monitors
// typical = normal dose rate from the ideal healthy run (µSv/h)
const SENSORS = [
  { id: "RAD-01", name: "Reactor cavity area monitor", type: "Gamma area monitor · ionisation chamber",
    where: "Operating deck, beside the reactor head", typical: 2.5, mount: "post", pos: [7.2, 12.3, 1.2] },
  { id: "RAD-02", name: "Primary loop area monitor", type: "Gamma area monitor · Geiger-Müller tube",
    where: "Steam generator compartment, next to a reactor coolant pump", typical: 8.0, mount: "post", pos: [5.2, 0, 7.4] },
  { id: "RAD-03", name: "Personnel airlock monitor", type: "Gamma area monitor · plastic scintillator",
    where: "Inside the personnel airlock entrance", typical: 0.8, mount: "wall", angle: -0.5 * Math.PI, height: 15.6 },
  { id: "RAD-04", name: "Plant vent stack monitor", type: "Effluent monitor · NaI(Tl) scintillation detector",
    where: "Vent stack sampling platform, outside containment", typical: 0.25, mount: "stack", pos: [24, 9, -24] },
  { id: "RAD-05", name: "Containment high-range monitor", type: "High-range gamma monitor · ionisation chamber",
    where: "Upper containment wall, below the dome", typical: 1.5, mount: "wall", angle: 0.78 * Math.PI, height: 24 },
];

const STATUS = {
  waiting: { color: 0x7d9a92, css: "#9aa9a4", label: "Awaiting data", speed: 0.6 },
  ok: { color: 0x5ff0a8, css: "#8ee2bd", label: "Healthy", speed: 1.0 },
  warn: { color: 0xffb54a, css: "#f3b969", label: "Attention", speed: 2.2 },
  bad: { color: 0xff4b40, css: "#ff746c", label: "Issue", speed: 4.0 },
};

function severity(issue) {
  if (!issue || issue === "Awaiting data") return "waiting";
  if (issue === "Healthy") return "ok";
  if (issue.startsWith("Watch") || issue.startsWith("At risk")) return "warn";
  return ["Missing readings", "Spike / outlier", "Stuck signal", "Noisy signal", "Drift high", "Drift low"].includes(issue) ? "bad" : "warn";
}

// ------------------------------------------------------------------ materials
const GAP = 0.56 * Math.PI;                   // cut-away opening, centred on the front (+z)
const OPEN = { start: GAP / 2, length: 2 * Math.PI - GAP };

function concreteTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#767b75";
  g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 5000; i++) {
    const v = 95 + Math.random() * 45;
    g.fillStyle = `rgba(${v},${v + 2},${v - 2},${Math.random() * 0.35})`;
    g.fillRect(Math.random() * 256, Math.random() * 256, 1 + Math.random() * 2, 1 + Math.random() * 2);
  }
  g.strokeStyle = "rgba(60,64,60,.35)";              // formwork joints
  for (let y = 0; y <= 256; y += 64) { g.beginPath(); g.moveTo(0, y); g.lineTo(256, y); g.stroke(); }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

const concreteMap = concreteTexture();
const mat = {
  concrete: (rx = 8, ry = 3) => {
    const map = concreteMap.clone();
    map.repeat.set(rx, ry);
    map.needsUpdate = true;
    return new THREE.MeshStandardMaterial({ map, roughness: 0.92, metalness: 0, side: THREE.DoubleSide });
  },
  liner: new THREE.MeshStandardMaterial({ color: 0x8c9894, roughness: 0.55, metalness: 0.6, side: THREE.DoubleSide }),
  steel: new THREE.MeshStandardMaterial({ color: 0xc3cbcf, roughness: 0.32, metalness: 0.9, side: THREE.DoubleSide }),
  paintedSteel: new THREE.MeshStandardMaterial({ color: 0xd9ddd6, roughness: 0.5, metalness: 0.35 }),
  pipe: new THREE.MeshStandardMaterial({ color: 0xb9c2c4, roughness: 0.3, metalness: 0.85 }),
  insulated: new THREE.MeshStandardMaterial({ color: 0xe4e6e0, roughness: 0.7, metalness: 0.15 }),
  floor: new THREE.MeshStandardMaterial({ color: 0x5f7f73, roughness: 0.8, metalness: 0.05 }),
  yellow: new THREE.MeshStandardMaterial({ color: 0xf2c230, roughness: 0.55, metalness: 0.2 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x2b3236, roughness: 0.6, metalness: 0.4 }),
  water: new THREE.MeshPhysicalMaterial({ color: 0x2f8fff, transparent: true, opacity: 0.28, roughness: 0.05,
    metalness: 0, transmission: 0, emissive: 0x0b4aa8, emissiveIntensity: 0.6, depthWrite: false }),
  fuel: new THREE.MeshStandardMaterial({ color: 0x9fe6ff, emissive: 0x3fb4ff, emissiveIntensity: 2.4, roughness: 0.4 }),
};

function mesh(geometry, material, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(geometry, material);
  m.position.set(x, y, z);
  m.castShadow = m.receiveShadow = true;
  return m;
}

// flat annular sector (a floor slab with a hole and the cut-away opening)
function slab(rIn, rOut, depth, start = OPEN.start, length = OPEN.length) {
  const s = new THREE.Shape(), steps = 72;
  for (let i = 0; i <= steps; i++) {
    const a = start + (length * i) / steps;
    s[i ? "lineTo" : "moveTo"](rOut * Math.sin(a), -rOut * Math.cos(a));
  }
  for (let i = steps; i >= 0; i--) {
    const a = start + (length * i) / steps;
    s.lineTo(rIn * Math.sin(a), -rIn * Math.cos(a));
  }
  const g = new THREE.ExtrudeGeometry(s, { depth, bevelEnabled: false, curveSegments: 4 });
  g.rotateX(-Math.PI / 2);
  return g;
}

// horizontal ring following the containment, interrupted by the cut-away
function arcTube(r, y, radius, material) {
  const pts = [];
  for (let i = 0; i <= 96; i++) {
    const a = OPEN.start + (OPEN.length * i) / 96;
    pts.push([Math.sin(a) * r, y, Math.cos(a) * r]);
  }
  return tube(pts, radius, material);
}

function tube(points, radius, material) {
  const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)), false, "catmullrom", 0.2);
  return mesh(new THREE.TubeGeometry(curve, 64, radius, 16, false), material);
}

// ------------------------------------------------------------------ plant
function buildPlant(scene) {
  const plant = new THREE.Group();
  scene.add(plant);

  // site and basemat
  const ground = mesh(new THREE.CircleGeometry(160, 64), new THREE.MeshStandardMaterial({ color: 0x1d2a26, roughness: 1 }), 0, -1.6, 0);
  ground.rotation.x = -Math.PI / 2;
  plant.add(ground);
  plant.add(mesh(new THREE.CylinderGeometry(21.5, 22, 1.6, 96), mat.concrete(12, 1), 0, -0.8, 0));
  plant.add(mesh(new THREE.CircleGeometry(19.2, 96).rotateX(-Math.PI / 2), mat.floor, 0, 0.01, 0));

  // containment: concrete wall + steel liner + hemispherical dome, cut open at the front
  const wallH = 28, rOut = 20, rIn = 19.2;
  plant.add(mesh(new THREE.CylinderGeometry(rOut, rOut, wallH, 128, 1, true, OPEN.start, OPEN.length), mat.concrete(10, 3), 0, wallH / 2, 0));
  plant.add(mesh(new THREE.CylinderGeometry(rIn, rIn, wallH, 128, 1, true, OPEN.start, OPEN.length), mat.liner, 0, wallH / 2, 0));
  const domeOut = new THREE.SphereGeometry(rOut, 128, 48, Math.PI / 2 + GAP / 2, OPEN.length, 0, Math.PI / 2);
  const domeIn = new THREE.SphereGeometry(rIn, 128, 48, Math.PI / 2 + GAP / 2, OPEN.length, 0, Math.PI / 2);
  plant.add(mesh(domeOut, mat.concrete(10, 2), 0, wallH, 0));
  plant.add(mesh(domeIn, mat.liner, 0, wallH, 0));
  // cut faces, so the 0.8 m wall thickness reads as solid concrete
  for (const a of [OPEN.start, OPEN.start + OPEN.length]) {
    const face = mesh(new THREE.BoxGeometry(rOut - rIn, wallH, 0.06), mat.concrete(1, 3), Math.sin(a) * (rIn + rOut) / 2, wallH / 2, Math.cos(a) * (rIn + rOut) / 2);
    face.rotation.y = a - Math.PI / 2;
    plant.add(face);
    const arc = new THREE.TorusGeometry((rIn + rOut) / 2, (rOut - rIn) / 2, 8, 64, Math.PI / 2);
    const rim = mesh(arc, mat.concrete(4, 1), 0, wallH, 0);
    rim.rotation.y = a - Math.PI / 2;                    // quarter arc up the dome's cut edge
    plant.add(rim);
  }
  // tendon gallery ring and polar crane
  plant.add(arcTube(18.6, 26, 0.35, mat.yellow));
  const bridge = mesh(new THREE.BoxGeometry(37, 1.1, 1.4), mat.yellow, 0, 26.6, -4);
  plant.add(bridge);
  plant.add(mesh(new THREE.BoxGeometry(2.4, 1.6, 3), mat.dark, -3, 27.4, -4));

  // operating deck and refuelling cavity walls (primary shield)
  plant.add(mesh(slab(5.2, rIn, 0.7), mat.concrete(6, 6), 0, 11.6, 0));
  plant.add(mesh(new THREE.CylinderGeometry(5.2, 5.2, 11.6, 96, 1, true, OPEN.start, OPEN.length), mat.concrete(3, 2), 0, 5.8, 0));
  plant.add(mesh(new THREE.CylinderGeometry(3.6, 3.6, 11.6, 96, 1, true, OPEN.start, OPEN.length), mat.concrete(3, 2), 0, 5.8, 0));
  plant.add(mesh(slab(3.6, 5.2, 0.05), mat.concrete(1, 1), 0, 11.6, 0));
  // deck railing along the opening
  for (const a of [OPEN.start, OPEN.start + OPEN.length]) {
    const rail = mesh(new THREE.BoxGeometry(0.08, 1.1, rIn - 5.2), mat.yellow, Math.sin(a) * 12.2, 12.85, Math.cos(a) * 12.2);
    rail.rotation.y = a;
    plant.add(rail);
  }

  plant.add(buildVessel());
  for (const side of [-1, 1]) plant.add(buildLoop(side));
  plant.add(buildPressurizer());
  plant.add(buildAirlock());
  plant.add(buildOutside());
  return plant;
}

function buildVessel() {
  const g = new THREE.Group();
  const r = 2.4, shellBottom = 2.6, shellTop = 10.6;
  const cut = { start: 0.42 * Math.PI, length: 2 * Math.PI - 0.84 * Math.PI };   // vessel cut-away shows the core
  g.add(mesh(new THREE.CylinderGeometry(r, r, shellTop - shellBottom, 96, 1, true, cut.start, cut.length), mat.steel, 0, (shellTop + shellBottom) / 2, 0));
  g.add(mesh(new THREE.SphereGeometry(r, 64, 24, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), mat.steel, 0, shellBottom, 0));
  g.add(mesh(new THREE.SphereGeometry(r * 1.02, 64, 24, 0, Math.PI * 2, 0, Math.PI / 2), mat.steel, 0, shellTop + 0.35, 0));
  g.add(mesh(new THREE.CylinderGeometry(r + 0.45, r + 0.45, 0.7, 96), mat.steel, 0, shellTop, 0));     // closure flange
  for (let i = 0; i < 36; i++) {                                                                         // head studs
    const a = (i / 36) * Math.PI * 2;
    g.add(mesh(new THREE.CylinderGeometry(0.07, 0.07, 1.0, 8), mat.dark, Math.sin(a) * (r + 0.3), shellTop + 0.4, Math.cos(a) * (r + 0.3)));
  }
  // control rod drive mechanisms on the head
  for (let ix = -2; ix <= 2; ix++) for (let iz = -2; iz <= 2; iz++) {
    if (ix * ix + iz * iz > 5) continue;
    g.add(mesh(new THREE.CylinderGeometry(0.11, 0.11, 3.2, 12), mat.paintedSteel, ix * 0.45, shellTop + 3.3, iz * 0.45));
  }
  g.add(mesh(new THREE.CylinderGeometry(1.5, 1.5, 0.25, 48), mat.dark, 0, shellTop + 4.9, 0));           // missile shield plate

  // coolant and the core: fuel assemblies glow with Cherenkov light
  g.add(mesh(new THREE.CylinderGeometry(r - 0.08, r - 0.08, shellTop - shellBottom, 64), mat.water, 0, (shellTop + shellBottom) / 2, 0));
  const pitch = 0.43, fuelGeo = new THREE.BoxGeometry(0.39, 3.8, 0.39), cells = [];
  for (let ix = -4; ix <= 4; ix++) for (let iz = -4; iz <= 4; iz++) if (Math.hypot(ix, iz) * pitch < 1.85) cells.push([ix * pitch, iz * pitch]);
  const fuel = new THREE.InstancedMesh(fuelGeo, mat.fuel, cells.length);
  const m4 = new THREE.Matrix4();
  cells.forEach(([x, z], i) => fuel.setMatrixAt(i, m4.makeTranslation(x, 5.1, z)));
  g.add(fuel);
  g.add(mesh(new THREE.CylinderGeometry(2.0, 2.0, 0.25, 48), mat.steel, 0, 3.05, 0));                     // core support plate
  g.add(mesh(new THREE.CylinderGeometry(2.0, 2.0, 0.25, 48), mat.steel, 0, 7.15, 0));                     // upper core plate
  for (let i = 0; i < 12; i++) {                                                                          // upper internals
    const a = (i / 12) * Math.PI * 2;
    g.add(mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.9, 8), mat.steel, Math.sin(a) * 1.3, 8.7, Math.cos(a) * 1.3));
  }
  const light = new THREE.PointLight(0x4fb6ff, 60, 14, 2);
  light.position.set(0, 5, 0);
  g.add(light);
  return g;
}

function buildLoop(side) {
  // one coolant loop: steam generator, reactor coolant pump, hot and cold legs, main steam line
  const g = new THREE.Group(), sx = side * 10.5, sz = -2.5;
  g.add(mesh(new THREE.SphereGeometry(1.75, 32, 16, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), mat.insulated, sx, 3, sz));
  g.add(mesh(new THREE.CylinderGeometry(1.75, 1.75, 10, 48), mat.insulated, sx, 8, sz));
  g.add(mesh(new THREE.CylinderGeometry(2.45, 1.75, 2.2, 48), mat.insulated, sx, 14.1, sz));
  g.add(mesh(new THREE.CylinderGeometry(2.45, 2.45, 5, 48), mat.insulated, sx, 17.7, sz));
  g.add(mesh(new THREE.SphereGeometry(2.45, 48, 16, 0, Math.PI * 2, 0, Math.PI / 2), mat.insulated, sx, 20.2, sz));
  for (const [dx, dz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {                                         // support columns
    g.add(mesh(new THREE.BoxGeometry(0.3, 2.4, 0.3), mat.yellow, sx + dx * 1.2, 1.2, sz + dz * 1.2));
  }
  for (const y of [9, 12]) g.add(mesh(new THREE.TorusGeometry(1.8, 0.09, 8, 48).rotateX(Math.PI / 2), mat.yellow, sx, y, sz));

  // reactor coolant pump with motor and flywheel
  const px = side * 6.6, pz = 4.8;
  g.add(mesh(new THREE.CylinderGeometry(0.95, 1.1, 2.2, 32), mat.steel, px, 2.7, pz));
  g.add(mesh(new THREE.CylinderGeometry(0.8, 0.8, 2.4, 32), mat.paintedSteel, px, 5.0, pz));
  g.add(mesh(new THREE.CylinderGeometry(1.0, 1.0, 0.35, 32), mat.dark, px, 6.4, pz));
  g.add(mesh(new THREE.BoxGeometry(2.6, 1.6, 2.6), mat.concrete(1, 1), px, 0.8, pz));

  g.add(tube([[side * 2.4, 8.4, 0], [side * 5.5, 8.4, -0.6], [sx - side * 0.6, 5, sz], [sx, 2.2, sz]], 0.48, mat.pipe));                  // hot leg
  g.add(tube([[sx, 1.6, sz], [sx - side * 1.8, 1.6, sz + 3.5], [px, 2.6, pz]], 0.42, mat.pipe));                                          // crossover leg
  g.add(tube([[px, 3.2, pz], [side * 4.2, 6, 3.2], [side * 2.4, 8.4, 1.4]], 0.42, mat.pipe));                                             // cold leg
  g.add(tube([[sx, 22.5, sz], [sx, 24.2, sz], [side * 16, 24.2, sz - 3], [side * 25, 24.2, sz - 6]], 0.45, mat.insulated));            // main steam line
  return g;
}

function buildPressurizer() {
  const g = new THREE.Group(), x = -3.5, z = -9.5;
  g.add(mesh(new THREE.CylinderGeometry(1.15, 1.15, 11, 40), mat.insulated, x, 11.5, z));
  g.add(mesh(new THREE.SphereGeometry(1.15, 32, 16), mat.insulated, x, 17, z));
  g.add(mesh(new THREE.SphereGeometry(1.15, 32, 16), mat.insulated, x, 6, z));
  g.add(tube([[x, 5, z], [x, 3, z + 1], [-5, 5.5, -1.8], [-7, 6.2, -1.6]], 0.22, mat.pipe));                    // surge line to hot leg
  return g;
}

function buildAirlock() {
  const g = new THREE.Group(), a = -0.58 * Math.PI, y = 13.6;
  const lock = mesh(new THREE.CylinderGeometry(1.5, 1.5, 6, 40), mat.steel, Math.sin(a) * 19.6, y, Math.cos(a) * 19.6);
  lock.rotation.set(Math.PI / 2, 0, 0);
  lock.rotation.y = a;
  lock.rotation.order = "YXZ";
  g.add(lock);
  const door = mesh(new THREE.CylinderGeometry(1.25, 1.25, 0.25, 40), mat.yellow, Math.sin(a) * 16.6, y, Math.cos(a) * 16.6);
  door.rotation.set(Math.PI / 2, 0, 0);
  door.rotation.y = a;
  door.rotation.order = "YXZ";
  g.add(door);
  return g;
}

function buildOutside() {
  const g = new THREE.Group();
  // auxiliary building and vent stack
  g.add(mesh(new THREE.BoxGeometry(30, 16, 14), mat.concrete(6, 3), 4, 8, -32));
  g.add(mesh(new THREE.BoxGeometry(30.5, 0.6, 14.5), mat.dark, 4, 16.2, -32));
  for (let i = 0; i < 7; i++) g.add(mesh(new THREE.BoxGeometry(2.2, 1.2, 0.1), new THREE.MeshStandardMaterial({ color: 0x9fb8c4, roughness: 0.2, metalness: 0.6 }), -8 + i * 4, 11, -24.94));
  g.add(mesh(new THREE.CylinderGeometry(0.9, 1.3, 46, 32), mat.paintedSteel, 24, 23, -24));
  for (const y of [8, 30]) g.add(mesh(new THREE.CylinderGeometry(2.2, 2.2, 0.25, 32), mat.yellow, 24, y, -24));
  g.add(tube([[19, 14, -28], [22, 15, -26], [23.2, 18, -24.6]], 0.5, mat.paintedSteel));
  // cooling tower in the distance
  const profile = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24, y = t * 46;
    profile.push(new THREE.Vector2(15 - 9 * Math.sin(Math.PI * Math.min(t, 0.82) / 1.64) + (t > 0.82 ? (t - 0.82) * 14 : 0), y));
  }
  const tower = mesh(new THREE.LatheGeometry(profile, 96), mat.concrete(12, 4), -48, 0, -46);
  g.add(tower);
  return g;
}

// ------------------------------------------------------------------ monitors
function faceplate(sensor) {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 160;
  const g = c.getContext("2d");
  g.fillStyle = "#e9ece6"; g.fillRect(0, 0, 256, 160);
  g.fillStyle = "#1b2226"; g.fillRect(14, 14, 150, 66);                     // display window
  g.fillStyle = "#f2c230"; g.beginPath(); g.arc(210, 47, 33, 0, Math.PI * 2); g.fill();   // trefoil
  g.fillStyle = "#151515";
  for (let k = 0; k < 3; k++) {
    const a = -Math.PI / 2 + k * (2 * Math.PI / 3);
    g.beginPath(); g.moveTo(210, 47); g.arc(210, 47, 28, a - 0.52, a + 0.52); g.closePath(); g.fill();
  }
  g.fillStyle = "#f2c230"; g.beginPath(); g.arc(210, 47, 8, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#151515"; g.beginPath(); g.arc(210, 47, 5, 0, Math.PI * 2); g.fill();
  g.font = "700 30px ui-monospace,monospace"; g.fillText(sensor.id, 14, 122);
  g.font = "16px system-ui,sans-serif"; g.fillStyle = "#4b5458"; g.fillText("RADIATION MONITOR", 14, 148);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function buildMonitor(sensor) {
  const root = new THREE.Group(), unit = new THREE.Group();
  root.add(unit);
  const face = new THREE.MeshStandardMaterial({ map: faceplate(sensor), roughness: 0.5 });
  const box = mesh(new THREE.BoxGeometry(1.3, 0.82, 0.5), [mat.paintedSteel, mat.paintedSteel, mat.paintedSteel, mat.paintedSteel, face, mat.paintedSteel], 0, 0, 0);
  unit.add(box);
  const screen = new THREE.MeshStandardMaterial({ color: 0x0b1210, emissive: STATUS.waiting.color, emissiveIntensity: 0.5 });
  unit.add(mesh(new THREE.PlaneGeometry(0.72, 0.32), screen, -0.2, 0.15, 0.255));
  unit.add(mesh(new THREE.CylinderGeometry(0.2, 0.2, 1.1, 24), mat.yellow, 0.25, -0.95, 0.05));        // detector probe
  unit.add(mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.12, 24), mat.dark, 0.25, -1.5, 0.05));
  const beaconMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: STATUS.waiting.color, emissiveIntensity: 2 });
  unit.add(mesh(new THREE.SphereGeometry(0.15, 20, 12), beaconMat, -0.45, 0.6, 0));
  unit.add(mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.1, 16), mat.dark, -0.45, 0.46, 0));

  if (sensor.mount === "wall") {
    const r = 19.05;
    root.position.set(Math.sin(sensor.angle) * r, sensor.height, Math.cos(sensor.angle) * r);
    root.rotation.y = sensor.angle + Math.PI;                                   // face the containment centre
    unit.position.z = 0.4;
    root.add(mesh(new THREE.BoxGeometry(0.9, 1.1, 0.35), mat.dark, 0, -0.2, 0.05));
  } else if (sensor.mount === "stack") {
    root.position.set(...sensor.pos);
    root.lookAt(0, sensor.pos[1], 0);
    unit.position.set(0, 1.9, 1.6);
    root.add(mesh(new THREE.BoxGeometry(0.15, 2.2, 0.15), mat.dark, 0, 0.8, 1.6));
  } else {
    root.position.set(...sensor.pos);
    root.lookAt(14, sensor.pos[1], 30);                                         // face the default camera
    unit.position.y = 2.6;
    root.add(mesh(new THREE.CylinderGeometry(0.1, 0.1, 2.2, 12), mat.dark, 0, 1.1, 0));
    root.add(mesh(new THREE.CylinderGeometry(0.45, 0.5, 0.12, 24), mat.dark, 0, 0.06, 0));
  }
  root.scale.setScalar(1.7);

  // halo ring shown when selected
  const halo = new THREE.Mesh(new THREE.TorusGeometry(1.25, 0.05, 8, 64), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 }));
  halo.visible = false;
  unit.add(halo);

  // floating label
  const el = document.createElement("button");
  el.className = "sensor-tag";
  el.type = "button";
  el.innerHTML = `<b>${sensor.id}</b><span>${sensor.name}</span><em>Awaiting data</em>`;
  const tag = new CSS2DObject(el);
  tag.position.set(0, 1.25, 0);
  unit.add(tag);

  root.traverse((o) => { if (o.isMesh) o.userData.sensor = sensor.id; });
  return { root, unit, screen, beaconMat, halo, el, state: "waiting", phase: Math.random() * 6 };
}

// ------------------------------------------------------------------ scene
function start() {
  const host = document.getElementById("scene3d"), panel = document.getElementById("sensorPanel");
  const monitors = new Map();
  let selected = null;

  renderPanel(panel, null, (id) => select(id));
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.9;
  host.prepend(renderer.domElement);

  const labels = new CSS2DRenderer();
  labels.domElement.className = "scene-labels";
  host.appendChild(labels.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1211);
  scene.fog = new THREE.Fog(0x0b1211, 90, 190);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  const camera = new THREE.PerspectiveCamera(38, 1, 0.5, 600);
  const home = { pos: new THREE.Vector3(26, 34, 82), target: new THREE.Vector3(0, 11, -2) };
  camera.position.copy(home.pos);

  scene.add(new THREE.HemisphereLight(0xcfe8ff, 0x1a2420, 0.55));
  const sun = new THREE.DirectionalLight(0xfff2dc, 1.7);
  sun.position.set(40, 70, 35);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -45, right: 45, top: 45, bottom: -45, near: 10, far: 180 });
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  const interior = new THREE.PointLight(0xdfeee8, 140, 45, 1.6);
  interior.position.set(-6, 22, -6);
  scene.add(interior);

  buildPlant(scene);
  for (const sensor of SENSORS) {
    const m = buildMonitor(sensor);
    m.el.addEventListener("click", () => select(sensor.id));
    scene.add(m.root);
    monitors.set(sensor.id, m);
  }

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(home.target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.minDistance = 8;
  controls.maxDistance = 150;
  controls.maxPolarAngle = Math.PI * 0.49;
  // keep the view inside the cut-away; idle camera sways between the limits
  controls.minAzimuthAngle = -0.42 * Math.PI;
  controls.maxAzimuthAngle = 0.42 * Math.PI;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.3;
  controls.addEventListener("start", () => { controls.autoRotate = false; fly = null; });

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.5, 0.45, 0.95);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  // click a monitor in the scene
  const ray = new THREE.Raycaster(), pointer = new THREE.Vector2();
  let downAt = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
    const r = renderer.domElement.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(pointer, camera);
    const hit = ray.intersectObjects([...monitors.values()].map((m) => m.root), true).find((h) => h.object.userData.sensor);
    if (hit) select(hit.object.userData.sensor);
  });
  renderer.domElement.addEventListener("pointermove", (e) => {
    const r = renderer.domElement.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(pointer, camera);
    const over = ray.intersectObjects([...monitors.values()].map((m) => m.root), true).some((h) => h.object.userData.sensor);
    renderer.domElement.style.cursor = over ? "pointer" : "grab";
  });

  // camera flight to a monitor
  let fly = null;
  function select(id) {
    selected = selected === id ? null : id;
    for (const [key, m] of monitors) {
      m.halo.visible = key === selected;
      m.el.classList.toggle("selected", key === selected);
    }
    panel.querySelectorAll(".sensor-card").forEach((c) => c.classList.toggle("selected", c.dataset.id === selected));
    controls.autoRotate = false;
    let target = home.target.clone(), pos = home.pos.clone();
    if (selected) {
      target = new THREE.Vector3();
      monitors.get(selected).unit.getWorldPosition(target);
      const away = new THREE.Vector3(target.x, 0, target.z);
      if (away.length() < 1) away.set(0, 0, 1);
      away.normalize();
      const inside = monitors.get(selected).root.position.length() < 19.5;
      // look at wall monitors from the middle of the containment, at others from outside
      pos = inside && SENSORS.find((s) => s.id === selected).mount === "wall"
        ? target.clone().addScaledVector(away, -13).add(new THREE.Vector3(0, 3, 0))
        : target.clone().addScaledVector(away, 12).add(new THREE.Vector3(4, 5, 9));
    }
    fly = { from: camera.position.clone(), to: pos, fromT: controls.target.clone(), toT: target, t: 0 };
  }

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h);
    labels.setSize(w, h);
    composer.setSize(w, h);
    bloom.resolution.set(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host);
  resize();
  if (host.clientWidth < 600) { home.pos.multiplyScalar(1.4); camera.position.copy(home.pos); }

  let visible = true;
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(host);

  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    if (!visible) return;
    const dt = Math.min(clock.getDelta(), 0.05), time = clock.elapsedTime;
    if (fly) {
      fly.t = Math.min(1, fly.t + dt / 1.2);
      const e = fly.t < 0.5 ? 4 * fly.t ** 3 : 1 - (-2 * fly.t + 2) ** 3 / 2;
      camera.position.lerpVectors(fly.from, fly.to, e);
      controls.target.lerpVectors(fly.fromT, fly.toT, e);
      if (fly.t === 1) fly = null;
    }
    for (const m of monitors.values()) {
      const s = STATUS[m.state], pulse = 0.5 + 0.5 * Math.sin(time * s.speed * 3 + m.phase);
      m.beaconMat.emissiveIntensity = m.state === "waiting" ? 0.8 : 1.5 + 3 * pulse;
      m.screen.emissiveIntensity = 0.35 + 0.25 * pulse;
      if (m.halo.visible) { m.halo.rotation.z = time; m.halo.material.opacity = 0.5 + 0.4 * pulse; }
    }
    if (controls.autoRotate) {
      const az = controls.getAzimuthalAngle();
      if (az > 0.3 * Math.PI) controls.autoRotateSpeed = Math.abs(controls.autoRotateSpeed);
      if (az < -0.15 * Math.PI) controls.autoRotateSpeed = -Math.abs(controls.autoRotateSpeed);
    }
    controls.update();
    composer.render();
    labels.render(scene, camera);
  });

  // results from the analysis in app.js
  function update(results) {
    const byId = new Map((results || []).map((r) => [r.sensor, r]));
    for (const sensor of SENSORS) {
      const m = monitors.get(sensor.id), r = byId.get(sensor.id);
      m.state = severity(r && r.issue);
      const s = STATUS[m.state];
      m.beaconMat.emissive.setHex(s.color);
      m.screen.emissive.setHex(s.color);
      m.el.style.setProperty("--tone", s.css);
      const act = r && r.maintenance && r.maintenance.action !== "None" ? ` · ${r.maintenance.action.toUpperCase()}` : "";
      m.el.querySelector("em").textContent = (r && Number.isFinite(r.latest) ? `${r.latest.toFixed(2)} · ${r.issue}` : r && r.issue !== "Awaiting data" ? r.issue : "Awaiting data") + act;
    }
    renderPanel(panel, byId, (id) => select(id), selected);
  }
  window.addEventListener("core-results", (e) => update(e.detail));
  update(window.coreResults);
  host.classList.add("ready");
}

function renderPanel(panel, byId, onSelect, selected = null) {
  panel.innerHTML = `<div class="eye">Radiation monitors</div>` + SENSORS.map((s) => {
    const r = byId && byId.get(s.id), st = STATUS[severity(r && r.issue)];
    const reading = r && Number.isFinite(r.latest) ? r.latest.toFixed(2) : "—";
    return `<button type="button" class="sensor-card${s.id === selected ? " selected" : ""}" data-id="${s.id}" style="--tone:${st.css}">
      <span class="sc-top"><b>${s.id}</b><i class="sc-scenario">${r ? r.issue : st.label}</i></span>
      <span class="sc-name">${s.name}</span>
      <span class="sc-meta">${s.type}</span>
      <span class="sc-meta">${s.where}</span>
      <span class="sc-read"><span>Latest <b>${reading}</b></span><span>Typical <b>${s.typical} µSv/h</b></span></span>
      ${r && r.maintenance && r.maintenance.action !== "None" ? `<span class="sc-action"><span class="act act-${r.maintenance.action.toLowerCase()}">${r.maintenance.action}</span><span class="urgency">${r.maintenance.urgency}</span></span>` : ""}
    </button>`;
  }).join("");
  panel.querySelectorAll(".sensor-card").forEach((c) => c.addEventListener("click", () => onSelect(c.dataset.id)));
}

try {
  start();
} catch (error) {
  const host = document.getElementById("scene3d");
  host.classList.add("no-webgl");
  host.insertAdjacentHTML("beforeend", `<div class="scene-fallback">The 3D model needs WebGL, which this browser has turned off. The monitor list still updates with your data.</div>`);
  const panel = document.getElementById("sensorPanel");
  const show = (results) => renderPanel(panel, new Map((results || []).map((r) => [r.sensor, r])), () => {});
  window.addEventListener("core-results", (e) => show(e.detail));
  show(window.coreResults);
  console.error(error);
}
