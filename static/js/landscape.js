// Frontier landscape: the smoothed frontier score over skeleton space, with FOSA's tasks as dots.
// Hovering (or tapping) a dot plays that task's trained policy. The page loads the classic-script
// bundle so it also works from file://; rebuild it after editing this file:
//   npx esbuild static/js/landscape.js --bundle --format=iife --minify --outfile=static/js/landscape.bundle.js
import * as THREE from './vendor/three/three.module.min.js';
import { OrbitControls } from './vendor/three/OrbitControls.js';

const section = document.querySelector('#landscape');
const panel = section.querySelector('.landscape');
const tabs = Array.from(section.querySelectorAll('[data-landscape-tabs] [role="tab"]'));
const view = section.querySelector('[data-landscape-view]');
const card = section.querySelector('[data-landscape-card]');
const video = card.querySelector('video');
const caption = card.querySelector('figcaption');
const status = section.querySelector('[data-landscape-status]');
const scriptUrl = document.currentScript.src;
const dataUrl = new URL('../data/', scriptUrl).href;
const videoUrl = new URL('../videos/landscape/', scriptUrl).href;
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const HEIGHT = 0.42, RADIUS = 0.0115, PICK_PX = 14;
// An Austin-sunset ramp after the CoRL 2026 logo: night teal, skyline, rose, burnt orange, sun.
const SUNSET = ['#203a43', '#4a8793', '#c98aa5', '#f08a3e', '#ffd966'].map(c => new THREE.Color(c));

function colormap(t) {
  const x = Math.min(1, Math.max(0, t)) * (SUNSET.length - 1), i = Math.min(SUNSET.length - 2, Math.floor(x));
  return SUNSET[i].clone().lerp(SUNSET[i + 1], x - i);
}

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
} catch {
  panel.hidden = false;
  status.textContent = 'This view needs WebGL.';
  throw new Error('WebGL unavailable');
}
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
view.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1.6, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(0, -1.55, 1.25);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.1);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 1.0;
controls.maxDistance = 3.2;
controls.maxPolarAngle = Math.PI * 0.47;
controls.autoRotate = !reduceMotion;
controls.autoRotateSpeed = 0.6;
scene.add(new THREE.HemisphereLight(0xffffff, 0x33281f, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(0.8, -1, 2);
scene.add(sun);

const world = new THREE.Group();
world.rotation.z = Math.PI / 4;  // diamond orientation, as in GFlowNet reward-landscape figures
scene.add(world);

let robot = 'jackal', dots = [], meshes = [], dotMesh = null, active = -1, pinned = false, pointerInside = false;
const matrix = new THREE.Matrix4(), point = new THREE.Vector3();

function load(name) {
  if (window.FOSA_LANDSCAPE && window.FOSA_LANDSCAPE[name]) return Promise.resolve(window.FOSA_LANDSCAPE[name]);
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = `${dataUrl}landscape-${name}.js`;
    script.onload = () => resolve(window.FOSA_LANDSCAPE[name]);
    script.onerror = reject;
    document.head.append(script);
  });
}

function clear() {
  meshes.forEach(mesh => { world.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); });
  meshes = [];
  dotMesh = null;
}

function build(data) {
  clear();
  const n = data.grid, h = data.height;
  const heightAt = (ix, row) => h[row * n + ix];

  // Surface: PlaneGeometry rows run from +y to -y, the data rows from y = 0 upward.
  const surface = new THREE.PlaneGeometry(1, 1, n - 1, n - 1);
  const positions = surface.attributes.position, colors = new Float32Array(positions.count * 3);
  for (let i = 0; i < positions.count; i++) {
    const value = heightAt(i % n, n - 1 - Math.floor(i / n));
    positions.setZ(i, value * HEIGHT);
    colormap(value).toArray(colors, i * 3);
  }
  surface.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  surface.computeVertexNormals();
  const skin = new THREE.Mesh(surface, new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.55, metalness: 0, side: THREE.DoubleSide, transparent: true, opacity: 0.94,
  }));

  // White iso-lines across the surface and around its border.
  const lines = [], step = Math.round((n - 1) / 6);
  const vertex = (ix, row) => [ix / (n - 1) - 0.5, row / (n - 1) - 0.5, heightAt(ix, row) * HEIGHT + 0.002];
  for (let k = 0; k < n; k += step) {
    for (let j = 0; j < n - 1; j++) {
      lines.push(...vertex(k, j), ...vertex(k, j + 1), ...vertex(j, k), ...vertex(j + 1, k));
    }
  }
  [[0, 0, 1, 0], [n - 1, 0, 0, 1], [0, n - 1, 1, 0], [0, 0, 0, 1]].forEach(([ix, row, dx, dy]) => {
    for (let j = 0; j < n - 1; j++) lines.push(...vertex(ix + dx * j, row + dy * j), ...vertex(ix + dx * (j + 1), row + dy * (j + 1)));
  });
  const grid = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(lines, 3)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 }),
  );

  // Dots, each with a slightly larger white back-face shell as an outline.
  dots = data.dots.map(([x, y, z, f, j0, j, key, description]) => ({
    position: new THREE.Vector3(x - 0.5, y - 0.5, z * HEIGHT + RADIUS * 0.6), f, j0, j, key, description,
  }));
  dotMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(RADIUS, 18, 12),
    new THREE.MeshStandardMaterial({ roughness: 0.35 }), dots.length);
  const shell = new THREE.InstancedMesh(new THREE.SphereGeometry(RADIUS * 1.35, 18, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.BackSide }), dots.length);
  dots.forEach((dot, i) => {
    matrix.makeTranslation(dot.position.x, dot.position.y, dot.position.z);
    dotMesh.setMatrixAt(i, matrix);
    shell.setMatrixAt(i, matrix);
    dotMesh.setColorAt(i, colormap(0.2 + 0.8 * dot.f));
  });
  meshes = [skin, grid, shell, dotMesh];
  meshes.forEach(mesh => world.add(mesh));
  active = -1;
  hideCard();
}

function screenOf(dot) {
  point.copy(dot.position).applyMatrix4(world.matrixWorld).project(camera);
  const rect = view.getBoundingClientRect();
  return { x: (point.x + 1) / 2 * rect.width, y: (1 - point.y) / 2 * rect.height, depth: point.z };
}

function nearest(clientX, clientY, tolerance) {
  const rect = view.getBoundingClientRect(), x = clientX - rect.left, y = clientY - rect.top;
  let best = -1, bestScore = Infinity;
  dots.forEach((dot, i) => {
    const s = screenOf(dot), d = Math.hypot(s.x - x, s.y - y);
    if (d < tolerance && d + s.depth * 4 < bestScore) { best = i; bestScore = d + s.depth * 4; }
  });
  return best;
}

function setScale(index, scale) {
  if (index < 0 || !dotMesh) return;
  const p = dots[index].position;
  matrix.makeScale(scale, scale, scale).setPosition(p.x, p.y, p.z);
  dotMesh.setMatrixAt(index, matrix);
  meshes[2].setMatrixAt(index, matrix);
  dotMesh.instanceMatrix.needsUpdate = meshes[2].instanceMatrix.needsUpdate = true;
}

function placeCard() {
  if (active < 0) return;
  const s = screenOf(dots[active]), rect = view.getBoundingClientRect();
  const w = card.offsetWidth, h = card.offsetHeight, gap = 16;
  let left = s.x + gap, top = s.y - h - gap;
  if (left + w > rect.width - 8) left = s.x - w - gap;
  if (top < 8) top = s.y + gap;
  card.style.left = `${Math.max(8, Math.min(rect.width - w - 8, left))}px`;
  card.style.top = `${Math.max(8, Math.min(rect.height - h - 8, top))}px`;
}

function showDot(index) {
  if (index === active) return;
  setScale(active, 1);
  active = index;
  setScale(active, 1.9);
  const dot = dots[index];
  caption.textContent = dot.description;
  video.src = `${videoUrl}${dot.key}.mp4`;
  video.play().catch(() => {});
  card.hidden = false;
  placeCard();
}

function hideCard() {
  setScale(active, 1);
  active = -1;
  pinned = false;
  card.hidden = true;
  video.pause();
  video.removeAttribute('src');
}

let down = null;
view.addEventListener('pointerdown', event => { down = { x: event.clientX, y: event.clientY }; });
view.addEventListener('pointermove', event => {
  pointerInside = true;
  if (event.buttons || pinned) return;
  const index = nearest(event.clientX, event.clientY, PICK_PX);
  view.classList.toggle('is-pointing', index >= 0);
  if (index >= 0) showDot(index);
  else if (active >= 0) hideCard();
});
view.addEventListener('pointerleave', () => { pointerInside = false; if (!pinned) hideCard(); });
view.addEventListener('pointerup', event => {
  if (!down || Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5) return;
  const index = nearest(event.clientX, event.clientY, event.pointerType === 'touch' ? 26 : PICK_PX);
  if (index < 0) { hideCard(); return; }
  if (pinned && index === active) { hideCard(); return; }
  showDot(index);
  pinned = true;
});
view.addEventListener('keydown', event => {
  if (!dots.length) return;
  const order = dots.map((_, i) => i).sort((a, b) => dots[b].f - dots[a].f);
  const at = order.indexOf(active);
  let next;
  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = order[(at + 1) % order.length];
  else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = order[(at - 1 + order.length) % order.length];
  else if (event.key === 'Escape') { hideCard(); return; }
  else return;
  event.preventDefault();
  showDot(next);
  pinned = true;
});

function resize() {
  const { width, height } = view.getBoundingClientRect();
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.position.setLength(width / height < 1.2 ? 2.8 : 2.3);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);

let visible = false;
new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(view);
renderer.setAnimationLoop(() => {
  if (!visible) return;
  controls.autoRotate = !reduceMotion && !pointerInside && !pinned && document.activeElement !== view;
  controls.update();
  renderer.render(scene, camera);
  placeCard();
});

function selectRobot(name, moveFocus = false) {
  robot = name;
  tabs.forEach(tab => {
    const on = tab.dataset.robot === name;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
    if (on) { panel.setAttribute('aria-labelledby', tab.id); if (moveFocus) tab.focus(); }
  });
  status.hidden = false;
  status.textContent = 'Loading…';
  load(name).then(data => {
    if (robot !== name) return;
    build(data);
    status.hidden = true;
  }).catch(() => { status.textContent = 'The landscape could not be loaded.'; });
}

tabs.forEach((tab, index) => {
  tab.id = `landscape-${tab.dataset.robot}-tab`;
  tab.addEventListener('click', () => selectRobot(tab.dataset.robot));
  tab.addEventListener('keydown', event => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    selectRobot(tabs[(index + step + tabs.length) % tabs.length].dataset.robot, true);
  });
});
section.querySelector('[data-landscape-tabs]').hidden = false;
panel.hidden = false;
resize();
selectRobot(robot);
