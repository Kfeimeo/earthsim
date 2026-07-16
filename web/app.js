/* EarthSim 前端: three.js 3D 地球 + WebSocket 实时数据 */

const THREE = await (async () => {
  const urls = [
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js",
    "https://unpkg.com/three@0.160.0/build/three.module.js",
  ];
  for (const u of urls) { try { return await import(u); } catch (e) { console.warn("three 加载失败:", u); } }
  document.body.innerHTML = "<p style='padding:40px'>无法加载 three.js(需要联网访问 CDN)。</p>";
  throw new Error("three.js unavailable");
})();

// ---------------- 图层定义(色标须与着色器一致) ----------------
const LAYERS = {
  none:   { label: "无叠加" },
  press:  { label: "气压",  unit: "hPa",  cm: 1 },
  temp:   { label: "气温",  unit: "°C",   cm: 2 },
  sst:    { label: "海温",  unit: "°C",   cm: 2, oceanOnly: true },
  hum:    { label: "水汽",  unit: "g/kg", cm: 3 },
  cloud:  { label: "云量",  unit: "",     cm: 4 },
  precip: { label: "降水",  unit: "mm/h", cm: 5 },
};
// JS 侧色标(画图例用), 与 GLSL 同步
const CM_STOPS = {
  1: [[0,[63,60,140]],[0.35,[80,140,200]],[0.5,[225,225,225]],[0.7,[240,170,80]],[1,[200,60,50]]],
  2: [[0,[40,40,150]],[0.3,[70,160,220]],[0.5,[120,210,160]],[0.7,[245,220,90]],[1,[220,50,40]]],
  3: [[0,[20,26,38]],[0.4,[50,140,150]],[0.75,[80,190,230]],[1,[230,245,255]]],
  4: [[0,[15,18,26]],[1,[245,247,250]]],
  5: [[0,[15,18,26]],[0.25,[70,200,220]],[0.6,[60,110,230]],[1,[190,80,230]]],
};

// ---------------- 场景 ----------------
const canvas = document.getElementById("globe");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
camera.position.set(0, 0.6, 3.2);

const meta = await (await fetch("/api/meta")).json();
const [NLAT, NLON] = meta.shape;
let atmosphereLevels = Array.isArray(meta.atmosphere_levels_m) ? meta.atmosphere_levels_m : [];
let windLayerAvailable = meta.wind_layer_available !== false;
let pendingWindLayer = null;
let oceanLayerAvailable = meta.ocean_layer_available === true;
let pendingOceanLayer = null;
document.getElementById("badge-mode").textContent = meta.mode === "live" ? "实时模拟" : "预演算回放";
document.getElementById("badge-backend").textContent = "后端 " + meta.backend.toUpperCase();

function setRecordingState(enabled) {
  const button = document.getElementById("btn-record");
  button.classList.toggle("recording", enabled);
  button.setAttribute("aria-pressed", String(enabled));
  button.title = enabled ? "\u505c\u6b62\u8bb0\u5f55" : "\u5f00\u59cb\u8bb0\u5f55";
}
if (meta.mode === "live") {
  setRecordingState(meta.recording !== false);
  document.getElementById("btn-reset").onclick = () => send({ cmd: "reset" });
  document.getElementById("btn-record").onclick = () => {
    send({ cmd: "record", enabled: !document.getElementById("btn-record").classList.contains("recording") });
  };
} else {
  document.querySelectorAll(".live-only").forEach(x => x.classList.add("hidden"));
}

const texLoader = new THREE.TextureLoader();
const baseTex = await texLoader.loadAsync("/api/basemap.png");
baseTex.colorSpace = THREE.SRGBColorSpace;
baseTex.minFilter = THREE.LinearMipmapLinearFilter;
baseTex.magFilter = THREE.LinearFilter;
baseTex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());

function makeDataTex(w, h) {
  const t = new THREE.DataTexture(new Uint8Array(w * h), w, h, THREE.RedFormat, THREE.UnsignedByteType);
  t.magFilter = t.minFilter = THREE.LinearFilter;
  t.wrapS = THREE.RepeatWrapping;
  t.needsUpdate = true;
  return t;
}
const dataTex = makeDataTex(NLON, NLAT);
const cloudTex = makeDataTex(NLON, NLAT);
const iceTex = makeDataTex(NLON, NLAT);
const landTex = makeDataTex(NLON, NLAT);
landTex.image.data.set(new Uint8Array(await (await fetch("/api/land.bin")).arrayBuffer()));
landTex.needsUpdate = true;

const uniforms = {
  baseTex: { value: baseTex }, dataTex: { value: dataTex },
  cloudTex: { value: cloudTex }, iceTex: { value: iceTex }, landTex: { value: landTex },
  cmap: { value: 0 }, oceanOnly: { value: 0 },
  showCloud: { value: 1.0 }, showIce: { value: 1.0 }, showNight: { value: 1.0 },
  sunDir: { value: new THREE.Vector3(1, 0, 0) },
};

const globeMat = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: /* glsl */`
    varying vec3 vN; varying vec3 vView;
    void main() {
      vN = normalize(position);
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vView = -mv.xyz;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */`
    precision highp float;
    varying vec3 vN; varying vec3 vView;
    uniform sampler2D baseTex, dataTex, cloudTex, iceTex, landTex;
    uniform int cmap, oceanOnly;
    uniform float showCloud, showIce, showNight;
    uniform vec3 sunDir;
    const float PI = 3.14159265;

    vec3 ramp(float t, vec3 c0, vec3 c1, vec3 c2, vec3 c3, vec3 c4, float p1, float p2, float p3) {
      if (t < p1) return mix(c0, c1, t / p1);
      if (t < p2) return mix(c1, c2, (t - p1) / (p2 - p1));
      if (t < p3) return mix(c2, c3, (t - p2) / (p3 - p2));
      return mix(c3, c4, (t - p3) / (1.0 - p3));
    }
    vec4 colormap(int id, float t) {
      if (id == 1) return vec4(ramp(t, vec3(.25,.24,.55), vec3(.31,.55,.78), vec3(.88,.88,.88), vec3(.94,.67,.31), vec3(.78,.24,.20), .35, .5, .7), 0.72);
      if (id == 2) return vec4(ramp(t, vec3(.16,.16,.59), vec3(.27,.63,.86), vec3(.47,.82,.63), vec3(.96,.86,.35), vec3(.86,.20,.16), .3, .5, .7), 0.70);
      if (id == 3) return vec4(ramp(t, vec3(.08,.10,.15), vec3(.20,.55,.59), vec3(.31,.75,.90), vec3(.90,.96,1.0), vec3(.90,.96,1.0), .4, .75, .95), 0.20 + 0.62 * t);
      if (id == 4) return vec4(vec3(0.96), 0.85 * t);
      if (id == 5) return vec4(ramp(t, vec3(.06,.07,.10), vec3(.27,.78,.86), vec3(.24,.43,.90), vec3(.75,.31,.90), vec3(.75,.31,.90), .25, .6, .9), smoothstep(0.02, 0.3, t) * 0.9);
      return vec4(0.0);
    }

    void main() {
      vec3 n = normalize(vN);
      float u = atan(n.x, n.z) / (2.0 * PI) + 0.5;
      float v = asin(clamp(n.y, -1.0, 1.0)) / PI + 0.5;
      vec2 uv = vec2(u, v);

      vec3 col = texture2D(baseTex, uv).rgb;

      // 冰雪覆盖
      float ice = texture2D(iceTex, uv).r;
      col = mix(col, vec3(0.88, 0.92, 0.97), showIce * ice * 0.85);

      // 数据图层
      if (cmap > 0) {
        float t = texture2D(dataTex, uv).r;
        vec4 c = colormap(cmap, t);
        float m = c.a;
        if (oceanOnly == 1) m *= 1.0 - step(0.5, texture2D(landTex, uv).r);
        col = mix(col, c.rgb, m);
      }
      // Linear opacity preserves a visible difference for every encoded cloud percent.
      float cl = texture2D(cloudTex, uv).r;
      float cloudOpacity = cl * 0.92;
      vec3 cloudColor = mix(vec3(0.64, 0.70, 0.77), vec3(0.985), cl);
      col = mix(col, cloudColor, showCloud * cloudOpacity);

      // Brighter direct sunlight without an additive surface halo.
      float d = dot(n, sunDir);
      float day = smoothstep(-0.16, 0.12, d);
      float illumination = 0.14 + day * 0.96 + max(d, 0.0) * 0.22;
      col *= mix(1.0, illumination, showNight);

      gl_FragColor = vec4(col, 1.0);
    }`,
});
const globe = new THREE.Mesh(new THREE.SphereGeometry(1, 160, 96), globeMat);
scene.add(globe);

// ---------------- 矢量叠加(风场/洋流) ----------------
function makeVecLines(color) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
  const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 });
  const l = new THREE.LineSegments(g, m);
  l.visible = false; scene.add(l);
  return l;
}
const windLines = makeVecLines(0xf5f7fa);
const oceanLines = makeVecLines(0x53d6ff);
const flowLines = { wind: windLines, ocean: oceanLines };
const flowScale = {
  wind: { vector: 0.0038, maxVector: 0.095, stream: 0.55, steps: 14 },
  ocean: { vector: 0.055, maxVector: 0.075, stream: 0.55, steps: 14 },
};
const initialStrides = meta.vector_strides || { wind: meta.vector_stride || 5, ocean: meta.vector_stride || 5 };
const flowState = {
  wind: { enabled: false, mode: "vector", stride: initialStrides.wind || 5, layer: meta.wind_layer_index || 0 },
  ocean: { enabled: false, mode: "vector", stride: initialStrides.ocean || 5, layer: meta.ocean_layer_index || 0 },
};

function lonLatToPos(latDeg, lonDeg, r) {
  const la = latDeg * Math.PI / 180, lo = (lonDeg / 360 - 0.5) * 2 * Math.PI;
  return [r * Math.cos(la) * Math.sin(lo), r * Math.sin(la), r * Math.cos(la) * Math.cos(lo)];
}
function updateVectors(lines, vec, shape, stride, scale, maxLen) {
  const [nla, nlo] = shape;
  const pts = [];
  for (let i = 0; i < nla; i++) {
    const lat = -90 + (i * stride + 0.5) * (180 / NLAT);
    if (Math.abs(lat) > 82) continue;
    for (let j = 0; j < nlo; j++) {
      const uu = vec[(i * nlo + j) * 2], vv = vec[(i * nlo + j) * 2 + 1];
      const sp = Math.hypot(uu, vv);
      if (sp < 0.02 * maxLen / scale) continue;
      const lon = j * stride * (360 / NLON);
      const p = lonLatToPos(lat, lon, 1.008);
      const la = lat * Math.PI / 180, lo = (lon / 360 - 0.5) * 2 * Math.PI;
      const east = [Math.cos(lo), 0, -Math.sin(lo)];
      const north = [-Math.sin(la) * Math.sin(lo), Math.cos(la), -Math.sin(la) * Math.cos(lo)];
      let L = Math.min(sp * scale, maxLen);
      const dir = [
        (east[0] * uu + north[0] * vv) / sp,
        (east[1] * uu + north[1] * vv) / sp,
        (east[2] * uu + north[2] * vv) / sp,
      ];
      const tip = [p[0] + dir[0] * L, p[1] + dir[1] * L, p[2] + dir[2] * L];
      pts.push(p[0], p[1], p[2], tip[0], tip[1], tip[2]);

      // 箭头头部：让“矢量图”和“流线图”在视觉上明确不同。
      const normal = lonLatToPos(lat, lon, 1);
      let side = [
        normal[1] * dir[2] - normal[2] * dir[1],
        normal[2] * dir[0] - normal[0] * dir[2],
        normal[0] * dir[1] - normal[1] * dir[0],
      ];
      const sideLen = Math.hypot(side[0], side[1], side[2]) || 1;
      side = [side[0] / sideLen, side[1] / sideLen, side[2] / sideLen];
      const headLen = Math.max(L * 0.32, 0.008);
      const headWid = headLen * 0.52;
      const base = [tip[0] - dir[0] * headLen, tip[1] - dir[1] * headLen, tip[2] - dir[2] * headLen];
      pts.push(
        tip[0], tip[1], tip[2], base[0] + side[0] * headWid, base[1] + side[1] * headWid, base[2] + side[2] * headWid,
        tip[0], tip[1], tip[2], base[0] - side[0] * headWid, base[1] - side[1] * headWid, base[2] - side[2] * headWid
      );
    }
  }
  lines.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
  lines.geometry.attributes.position.needsUpdate = true;
}
function clearLines(lines) {
  lines.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
  lines.geometry.attributes.position.needsUpdate = true;
}
function sampleVec(vec, shape, stride, lat, lon) {
  const [nla, nlo] = shape;
  const i = Math.max(0, Math.min(nla - 1, Math.round((lat + 90) / (180 / NLAT) / stride)));
  const jj = Math.round((((lon % 360) + 360) % 360) / (360 / NLON) / stride);
  const j = ((jj % nlo) + nlo) % nlo;
  const k = (i * nlo + j) * 2;
  return [vec[k], vec[k + 1]];
}
function updateStreamlines(lines, vec, shape, stride, stepScale, steps) {
  const [nla, nlo] = shape;
  const pts = [];
  const dLat = 180 / NLAT;
  const dLon = 360 / NLON;
  const seedStep = stride <= 2 ? 4 : stride <= 4 ? 3 : stride <= 7 ? 2 : 1;
  const stepDeg = Math.max(dLat, dLon) * stride * stepScale;

  for (let i = 0; i < nla; i += seedStep) {
    let lat = -90 + (i * stride + 0.5) * dLat;
    if (Math.abs(lat) > 82) continue;
    for (let j = 0; j < nlo; j += seedStep) {
      let la = lat;
      let lo = j * stride * dLon;
      for (let s = 0; s < steps; s++) {
        const [uu, vv] = sampleVec(vec, shape, stride, la, lo);
        const sp = Math.hypot(uu, vv);
        if (sp < 1e-5) break;
        const cosLat = Math.max(Math.cos(la * Math.PI / 180), 0.2);
        const nextLat = Math.max(-82, Math.min(82, la + (vv / sp) * stepDeg));
        const nextLon = (lo + (uu / sp) * stepDeg / cosLat + 360) % 360;
        const p0 = lonLatToPos(la, lo, 1.008);
        const p1 = lonLatToPos(nextLat, nextLon, 1.008);
        pts.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);
        la = nextLat; lo = nextLon;
      }
    }
  }
  lines.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
  lines.geometry.attributes.position.needsUpdate = true;
}
function updateFlow(name, m) {
  const state = flowState[name];
  const lines = flowLines[name];
  lines.visible = state.enabled;
  if (!state.enabled || !m?.vecf?.[name]) return;
  const vf = m.vecf[name];
  const scale = flowScale[name];
  if (state.mode === "streamline") {
    updateStreamlines(lines, vf.data, vf.shape, vf.stride, scale.stream, scale.steps);
  } else {
    updateVectors(lines, vf.data, vf.shape, vf.stride, scale.vector, scale.maxVector);
  }
}

// ---------------- 相机控制(拖动/缩放/惯性) ----------------
let theta = 0.9, phi = 0.35, dist = 3.2, vTh = 0, vPh = 0;
let dragging = false, lastX = 0, lastY = 0, moved = 0;
function applyCamera() {
  phi = Math.max(-1.45, Math.min(1.45, phi));
  dist = Math.max(1.35, Math.min(8, dist));
  camera.position.set(dist * Math.cos(phi) * Math.sin(theta), dist * Math.sin(phi), dist * Math.cos(phi) * Math.cos(theta));
  camera.lookAt(0, 0, 0);
}

// ---------------- 区域编辑(笔刷) ----------------
let editMode = false, painting = false, lastPaint = 0;
const brush = new THREE.Mesh(new THREE.RingGeometry(0.92, 1, 48),
  new THREE.MeshBasicMaterial({ color: 0xff7043, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false }));
brush.visible = false; scene.add(brush);

const editKind = () => document.getElementById("edit-kind").value;
const editDelta = () => parseFloat(document.getElementById("edit-delta").value);
const editRadius = () => parseFloat(document.getElementById("edit-radius").value);
const cycloneStrength = () => parseFloat(document.getElementById("cyclone-strength").value);

function updateEditLabels() {
  const kind = editKind();
  const d = editDelta();
  document.getElementById("edit-delta-label").textContent = (d > 0 ? "+" : "") + d + "°C";
  document.getElementById("edit-radius-label").textContent = editRadius() + "km";
  document.getElementById("cyclone-strength-label").textContent = cycloneStrength() + "m/s";
  document.getElementById("edit-delta").closest(".edit-row").classList.toggle("hidden", kind !== "temp");
  document.getElementById("edit-target").closest(".edit-row").classList.toggle("hidden", kind !== "temp");
  document.getElementById("cyclone-strength-row").classList.toggle("hidden", kind !== "cyclone");
  if (kind === "wind_zero") brush.material.color.set(0x53a8ff);
  else if (kind === "cyclone") brush.material.color.set(0xf5b14c);
  else brush.material.color.set(d >= 0 ? 0xff7043 : 0x53a8ff);
}
if (meta.mode === "live") {
  document.getElementById("edit-panel").classList.remove("hidden");
  document.getElementById("btn-edit").onclick = () => {
    editMode = !editMode;
    document.getElementById("btn-edit").classList.toggle("active", editMode);
    document.getElementById("btn-edit").textContent = editMode ? "🖌 退出编辑模式" : "🖌 进入编辑模式";
    canvas.classList.toggle("editing", editMode);
    brush.visible = false; painting = false;
    document.getElementById("hint").textContent = editMode
      ? "点击/按住拖动 = 区域编辑 · 滚轮缩放 · 退出编辑后可旋转"
      : "拖动旋转 · 滚轮缩放 · 点击地表查看天气";
  };
  document.getElementById("edit-kind").onchange = updateEditLabels;
  document.getElementById("edit-delta").oninput = updateEditLabels;
  document.getElementById("edit-radius").oninput = updateEditLabels;
  document.getElementById("cyclone-strength").oninput = updateEditLabels;
  updateEditLabels();
}
function hitGlobe(e) {
  const r = canvas.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(
    ((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  return raycaster.intersectObject(globe)[0] || null;
}
function moveBrush(hit) {
  const n = hit.point.clone().normalize();
  const ang = editRadius() / 6371;                 // 弧度 ≈ 球面半径
  brush.position.copy(n.clone().multiplyScalar(1.006));
  brush.lookAt(0, 0, 0);
  brush.scale.setScalar(Math.max(Math.sin(ang), 0.008));
  brush.visible = true;
  return n;
}
function paintAt(n) {
  const now = performance.now();
  if (now - lastPaint < 90) return;                 // 节流
  lastPaint = now;
  const lat = Math.asin(n.y) * 180 / Math.PI;
  const lon = ((Math.atan2(n.x, n.z) / (2 * Math.PI) + 0.5) * 360) % 360;
  const kind = editKind();
  if (kind === "wind_zero") {
    send({ cmd: "edit_wind_zero", lat, lon, radius: editRadius(), layer: flowState.wind.layer });
  } else if (kind === "cyclone") {
    send({ cmd: "edit_cyclone", lat, lon, radius: editRadius(),
           strength: cycloneStrength(), layer: flowState.wind.layer });
  } else {
    send({ cmd: "edit_temp", lat, lon, radius: editRadius(),
           delta: editDelta(), target: document.getElementById("edit-target").value });
  }
}

canvas.addEventListener("pointerdown", e => {
  if (editMode) {
    const hit = hitGlobe(e);
    if (hit) { painting = true; lastPaint = 0; paintAt(moveBrush(hit)); }
    canvas.setPointerCapture(e.pointerId);
    return;
  }
  dragging = true; moved = 0; lastX = e.clientX; lastY = e.clientY;
  canvas.classList.add("dragging"); canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", e => {
  if (editMode) {
    const hit = hitGlobe(e);
    if (hit) { const n = moveBrush(hit); if (painting) paintAt(n); }
    else brush.visible = false;
    return;
  }
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  moved += Math.abs(dx) + Math.abs(dy);
  vTh = -dx * 0.005; vPh = dy * 0.005;
  theta += vTh; phi += vPh;
  lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener("pointerup", e => {
  if (editMode) { painting = false; return; }
  dragging = false; canvas.classList.remove("dragging");
  if (moved < 6) pick(e);
});
canvas.addEventListener("wheel", e => { e.preventDefault(); dist *= 1 + Math.sign(e.deltaY) * 0.08; }, { passive: false });

// ---------------- 点击分析 ----------------
const raycaster = new THREE.Raycaster();
const marker = new THREE.Mesh(new THREE.RingGeometry(0.015, 0.024, 24),
  new THREE.MeshBasicMaterial({ color: 0xf5b14c, side: THREE.DoubleSide }));
marker.visible = false; scene.add(marker);

let selectedPoint = null;
let analysisSeq = 0;
let lastAnalysisAt = 0;
let analysisPending = false;

function pick(e) {
  const r = canvas.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(
    ((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = raycaster.intersectObject(globe)[0];
  if (!hit) return;
  const n = hit.point.clone().normalize();
  const lat = Math.asin(n.y) * 180 / Math.PI;
  const lon = ((Math.atan2(n.x, n.z) / (2 * Math.PI) + 0.5) * 360) % 360;
  marker.position.copy(n.multiplyScalar(1.012));
  marker.lookAt(0, 0, 0); marker.visible = true;
  selectedPoint = { lat, lon };
  refreshSelectedAnalysis(true);
}
async function refreshSelectedAnalysis(force = false) {
  if (!selectedPoint || analysisPending) return;
  const now = performance.now();
  if (!force && now - lastAnalysisAt < 120) return;
  lastAnalysisAt = now;
  analysisPending = true;
  const seq = ++analysisSeq;
  const { lat, lon } = selectedPoint;
  try {
    const d = await (await fetch(`/api/analyze?lat=${lat.toFixed(2)}&lon=${lon.toFixed(2)}`)).json();
    if (seq === analysisSeq && selectedPoint) showCard(d);
  } catch (err) {
    console.warn("analyze failed", err);
  } finally {
    analysisPending = false;
  }
}
function showCard(d) {
  if (d.error) return;
  document.getElementById("card-icon").textContent = d.icon;
  document.getElementById("card-weather").textContent = d.weather;
  document.getElementById("card-coord").textContent =
    `${Math.abs(d.lat).toFixed(1)}°${d.lat >= 0 ? "N" : "S"}  ${d.lon.toFixed(1)}°E · ${d.surface}`;
  const rows = [["气温", d.temp + " °C"], ["气压", d.pressure + " hPa"],
    ["比湿", d.humidity + " g/kg"], ["云量", d.cloud + " %"],
    ["降水", d.precip + " mm/h"], ["风", d.wind_speed + " m/s / " + d.wind_dir + "°"]];
  if (d.ground_water !== undefined) rows.push(["地表储水", d.ground_water + " mm"]);
  if (d.sst !== undefined) rows.push(["海温", d.sst + " °C"], ["洋流", d.current + " m/s"]);
  if (d.ice > 0) rows.push(["冰雪", d.ice + " %"]);
  document.getElementById("card-grid").innerHTML =
    rows.map(([k, v]) => `<span class="k">${k}</span><span class="v">${v}</span>`).join("");
  document.getElementById("analysis-card").classList.remove("hidden");
}
document.getElementById("card-close").onclick = () => {
  document.getElementById("analysis-card").classList.add("hidden");
  selectedPoint = null;
  analysisSeq++;
  marker.visible = false;
};

// ---------------- 图层 UI ----------------
let activeLayer = "temp";
const list = document.getElementById("layer-list");
for (const [k, def] of Object.entries(LAYERS)) {
  const b = document.createElement("button");
  b.className = "layer-btn" + (k === activeLayer ? " active" : "");
  b.textContent = def.label; b.dataset.k = k;
  b.onclick = () => { activeLayer = k; document.querySelectorAll(".layer-btn").forEach(x => x.classList.toggle("active", x.dataset.k === k)); refreshLayer(); };
  list.appendChild(b);
}
const toggles = { cloud: "showCloud", ice: "showIce", night: "showNight" };
for (const [id, u] of Object.entries(toggles))
  document.getElementById("tg-" + id).onchange = e => { uniforms[u].value = e.target.checked ? 1 : 0; };
function strideToDensity(stride) {
  return Math.max(1, Math.min(10, 12 - parseInt(stride || 5)));
}
function densityToStride(density) {
  return Math.max(2, 12 - parseInt(density || 7));
}
function windLayerCount() {
  return Math.max(atmosphereLevels.length, 1);
}
function clampWindLayer(k) {
  return Math.max(0, Math.min(windLayerCount() - 1, parseInt(k || 0)));
}
function oceanLayerCount() {
  return oceanLayerAvailable ? 2 : 1;
}
function clampOceanLayer(k) {
  return Math.max(0, Math.min(oceanLayerCount() - 1, parseInt(k || 0)));
}
function formatWindLayerLabel(k) {
  const z = atmosphereLevels[k];
  return Number.isFinite(z) ? `${Math.round(z)} m` : `第 ${k + 1} 层`;
}
function formatOceanLayerLabel(k) {
  return k === 1 ? "深层" : "表层";
}
function sameAtmosphereLevels(next) {
  if (!Array.isArray(next) || next.length !== atmosphereLevels.length) return false;
  return next.every((z, i) => z === atmosphereLevels[i]);
}
function setupWindLayerControl() {
  const row = document.getElementById("wind-layer-row");
  const layer = document.getElementById("wind-layer");
  const count = windLayerCount();
  layer.innerHTML = "";
  for (let k = 0; k < count; k++) {
    const opt = document.createElement("option");
    opt.value = String(k);
    opt.textContent = formatWindLayerLabel(k);
    layer.appendChild(opt);
  }
  row.classList.toggle("hidden", count <= 1 || !windLayerAvailable);
  layer.disabled = !windLayerAvailable;
  flowState.wind.layer = clampWindLayer(flowState.wind.layer);
  if (pendingWindLayer !== null) pendingWindLayer = clampWindLayer(pendingWindLayer);
  layer.value = String(flowState.wind.layer);
}
function setupOceanLayerControl() {
  const row = document.getElementById("ocean-layer-row");
  const layer = document.getElementById("ocean-layer");
  const count = oceanLayerCount();
  layer.innerHTML = "";
  for (let k = 0; k < count; k++) {
    const opt = document.createElement("option");
    opt.value = String(k);
    opt.textContent = formatOceanLayerLabel(k);
    layer.appendChild(opt);
  }
  row.classList.toggle("hidden", count <= 1 || !oceanLayerAvailable);
  layer.disabled = !oceanLayerAvailable;
  flowState.ocean.layer = clampOceanLayer(flowState.ocean.layer);
  if (pendingOceanLayer !== null) pendingOceanLayer = clampOceanLayer(pendingOceanLayer);
  layer.value = String(flowState.ocean.layer);
}
function syncWindLayerMeta(m) {
  let controlsChanged = false;
  if (Array.isArray(m.atmosphere_levels_m) && !sameAtmosphereLevels(m.atmosphere_levels_m)) {
    atmosphereLevels = m.atmosphere_levels_m;
    controlsChanged = true;
  }
  if (typeof m.wind_layer_available === "boolean" && windLayerAvailable !== m.wind_layer_available) {
    windLayerAvailable = m.wind_layer_available;
    controlsChanged = true;
  }
  if (!windLayerAvailable) pendingWindLayer = null;
  if (controlsChanged) setupWindLayerControl();
}
function syncWindLayerSelection(m) {
  if (!Number.isInteger(m.wind_layer_index)) return true;
  const serverLayer = clampWindLayer(m.wind_layer_index);
  const waitingForSelection = pendingWindLayer !== null;
  if (waitingForSelection && serverLayer !== pendingWindLayer) return false;

  pendingWindLayer = null;
  flowState.wind.layer = serverLayer;
  const layer = document.getElementById("wind-layer");
  if (layer) layer.value = String(serverLayer);
  return true;
}
function syncOceanLayerMeta(m) {
  if (typeof m.ocean_layer_available === "boolean" && oceanLayerAvailable !== m.ocean_layer_available) {
    oceanLayerAvailable = m.ocean_layer_available;
    if (!oceanLayerAvailable) pendingOceanLayer = null;
    setupOceanLayerControl();
  }
}
function syncOceanLayerSelection(m) {
  if (!Number.isInteger(m.ocean_layer_index)) return true;
  const serverLayer = clampOceanLayer(m.ocean_layer_index);
  const waitingForSelection = pendingOceanLayer !== null;
  if (waitingForSelection && serverLayer !== pendingOceanLayer) return false;

  pendingOceanLayer = null;
  flowState.ocean.layer = serverLayer;
  const layer = document.getElementById("ocean-layer");
  if (layer) layer.value = String(serverLayer);
  return true;
}
function setupFlowControls(name) {
  const tg = document.getElementById("tg-" + name);
  const mode = document.getElementById(name + "-mode");
  const density = document.getElementById(name + "-density");
  const label = document.getElementById(name + "-density-label");
  const layer = document.getElementById(name + "-layer");
  const state = flowState[name];

  density.value = strideToDensity(state.stride);
  const updateDensityLabel = () => { label.textContent = density.value; };
  updateDensityLabel();

  tg.onchange = e => {
    state.enabled = e.target.checked;
    if (!state.enabled) clearLines(flowLines[name]);
    updateFlow(name, lastFrame);
  };
  mode.onchange = e => {
    state.mode = e.target.value;
    updateFlow(name, lastFrame);
  };
  if (layer) {
    layer.onchange = e => {
      state.layer = name === "wind" ? clampWindLayer(e.target.value) : clampOceanLayer(e.target.value);
      if (name === "wind") pendingWindLayer = state.layer;
      else pendingOceanLayer = state.layer;
      layer.value = String(state.layer);
      clearLines(flowLines[name]);
      send({ cmd: name === "wind" ? "set_wind_layer" : "set_ocean_layer", value: state.layer });
    };
  }
  density.oninput = () => {
    state.stride = densityToStride(density.value);
    updateDensityLabel();
    send({ cmd: "set_vector_stride", target: name, value: state.stride });
  };
}
function syncFlowSettings() {
  for (const [name, state] of Object.entries(flowState)) {
    send({ cmd: "set_vector_stride", target: name, value: state.stride });
  }
  send({ cmd: "set_wind_layer", value: flowState.wind.layer });
  send({ cmd: "set_ocean_layer", value: flowState.ocean.layer });
}
setupWindLayerControl();
setupOceanLayerControl();
setupFlowControls("wind");
setupFlowControls("ocean");

let lastFrame = null;
function refreshLayer() {
  const def = LAYERS[activeLayer];
  uniforms.cmap.value = def.cm || 0;
  uniforms.oceanOnly.value = def.oceanOnly ? 1 : 0;
  if (lastFrame && def.cm) {
    const L = lastFrame.layers.find(l => l.name === activeLayer);
    dataTex.image.data.set(lastFrame.bytes[activeLayer]);
    dataTex.needsUpdate = true;
    drawLegend(def, L);
  } else drawLegend(def, null);
}
function drawLegend(def, L) {
  const c = document.getElementById("legend-bar").getContext("2d");
  const g = c.createLinearGradient(0, 0, 160, 0);
  if (def.cm) for (const [p, [r, gg, b]] of CM_STOPS[def.cm]) g.addColorStop(p, `rgb(${r},${gg},${b})`);
  else g.addColorStop(0, "#222"), g.addColorStop(1, "#222");
  c.fillStyle = g; c.fillRect(0, 0, 160, 10);
  document.getElementById("legend-name").textContent = def.cm ? def.label + (def.unit ? ` (${def.unit})` : "") : "";
  document.getElementById("legend-min").textContent = L ? L.min : "";
  document.getElementById("legend-max").textContent = L ? L.max : "";
}

// ---------------- WebSocket ----------------
const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
ws.binaryType = "arraybuffer";
const send = o => ws.readyState === 1 && ws.send(JSON.stringify(o));
ws.onopen = syncFlowSettings;

ws.onmessage = ev => {
  const dv = new DataView(ev.data);
  const jlen = dv.getUint32(0, true);
  const m = JSON.parse(new TextDecoder().decode(new Uint8Array(ev.data, 4, jlen)));
  const base = 4 + jlen;
  m.bytes = {};
  for (const L of m.layers) m.bytes[L.name] = new Uint8Array(ev.data.slice(base + L.off, base + L.off + L.len));
  m.vecf = {};
  for (const V of m.vecs) m.vecf[V.name] = { data: new Float32Array(ev.data.slice(base + V.off, base + V.off + V.len)), shape: V.shape, stride: V.stride };
  syncWindLayerMeta(m);
  syncOceanLayerMeta(m);
  const windLayerMatchesSelection = syncWindLayerSelection(m);
  const oceanLayerMatchesSelection = syncOceanLayerSelection(m);
  lastFrame = m;
  render(m, windLayerMatchesSelection, oceanLayerMatchesSelection);
};

function render(m, windLayerMatchesSelection = true, oceanLayerMatchesSelection = true) {
  cloudTex.image.data.set(m.bytes.cloud); cloudTex.needsUpdate = true;
  iceTex.image.data.set(m.bytes.ice); iceTex.needsUpdate = true;
  if (LAYERS[activeLayer].cm) {
    dataTex.image.data.set(m.bytes[activeLayer]); dataTex.needsUpdate = true;
    drawLegend(LAYERS[activeLayer], m.layers.find(l => l.name === activeLayer));
  }
  if (windLayerMatchesSelection) updateFlow("wind", m);
  if (oceanLayerMatchesSelection) updateFlow("ocean", m);
  refreshSelectedAnalysis();
  // 太阳方向
  const [sla, slo] = m.subsolar;
  const p = lonLatToPos(sla, slo, 1);
  uniforms.sunDir.value.set(p[0], p[1], p[2]);
  // 时钟与控制状态
  document.getElementById("sim-clock").textContent = m.time.replace("T", "  ").slice(0, 17) + " UTC";
  document.getElementById("btn-play").textContent = m.playing ? "⏸" : "▶";
  playing = m.playing;
  if (m.mode === "live" && typeof m.recording === "boolean") setRecordingState(m.recording);
  if (m.mode === "playback") {
    document.querySelectorAll(".pb-only").forEach(x => x.classList.remove("hidden"));
    const tl = document.getElementById("timeline");
    tl.max = m.nframes - 1;
    if (!seeking) tl.value = m.frame;
    document.getElementById("frame-label").textContent = `${m.frame + 1}/${m.nframes}`;
  }
}

// ---------------- 播放控制 ----------------
let playing = true, seeking = false;
document.getElementById("btn-play").onclick = () => send({ cmd: playing ? "pause" : "play" });
document.getElementById("btn-step").onclick = () => send({ cmd: "step" });
document.getElementById("btn-back").onclick = () => send({ cmd: "back" });
const speedEl = document.getElementById("speed");
speedEl.oninput = () => {
  const s = Math.pow(2, parseFloat(speedEl.value));
  document.getElementById("speed-label").textContent = "×" + (s < 1 ? s.toFixed(2) : s.toFixed(1));
  send({ cmd: "speed", value: s });
};
const tl = document.getElementById("timeline");
tl.addEventListener("pointerdown", () => seeking = true);
tl.addEventListener("change", () => { send({ cmd: "seek", value: parseInt(tl.value) }); seeking = false; });
window.addEventListener("keydown", e => {
  if (e.key === " ") { e.preventDefault(); send({ cmd: playing ? "pause" : "play" }); }
  if (e.key === "ArrowRight") send({ cmd: "step" });
  if (e.key === "ArrowLeft") send({ cmd: "back" });
});

// ---------------- 主循环 ----------------
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener("resize", resize); resize();
refreshLayer();
(function loop() {
  requestAnimationFrame(loop);
  if (!dragging) { theta += vTh; phi += vPh; vTh *= 0.93; vPh *= 0.93; }
  applyCamera();
  renderer.render(scene, camera);
})();
