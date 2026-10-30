// A small Three.js view of a molecule with its polarizability ellipsoids, as
// the legacy JSmol view drew them. Only the atoms are shown: a molecule has no
// cell (the box matsci-parse may put it in is not drawn).

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { cartesian, covalentRadii } from "matsci-parse";
import { createMatrix, eigh } from "matsci-parse/matrix";

// The legacy app's colours
const COLORS = { H: 0xc0c0c0, C: 0x505050, N: 0x0000cc, O: 0xcc0000, S: 0xffc832, Cl: 0x00cc00 };
const BOND_FACTOR = 1.1; // bonded below 1.1 (r_i + r_j)

/** Semi-axes of the ellipsoid of a 3x3 tensor (row-major): the eigenvectors,
 * scaled by sqrt(|eigenvalue|) (as the legacy view), with the signs. */
function ellipsoidAxes(tensor) {
  const symmetric = tensor.map((_, k) => {
    const [i, j] = [Math.floor(k / 3), k % 3];
    return (tensor[3 * i + j] + tensor[3 * j + i]) / 2;
  });
  const { values, vectors } = eigh(createMatrix(3, 3, symmetric));
  return [0, 1, 2].map((c) => ({
    direction: new THREE.Vector3(vectors.data[c], vectors.data[3 + c], vectors.data[6 + c]),
    length: Math.sqrt(Math.abs(values[c])),
    positive: values[c] >= 0,
  }));
}

/** An HTML label next to an atom (its element and number, as in the table). */
function atomLabel(text, position, radius) {
  const div = document.createElement("div");
  div.textContent = text;
  div.className = "rounded bg-white/80 px-1 text-[11px] font-medium leading-tight text-slate-800";
  const label = new CSS2DObject(div);
  label.position.copy(position).add(new THREE.Vector3(radius, radius, 0));
  return label;
}

/** The objects of the scene, centred on the molecule, and its radius. The
 * labels and the ellipsoids (with their axes) are listed apart, to be shown
 * or hidden without rebuilding the scene. */
function buildMolecule(structure, tensors, scale, showAxes) {
  const group = new THREE.Group();
  const labels = [];
  const ellipsoids = [];
  const symbols = structure.sites.map((site) => site.species.symbol);
  const positions = structure.sites.map((site) => new THREE.Vector3(...cartesian(structure.lattice, site)));
  const center = positions.reduce((c, p) => c.add(p), new THREE.Vector3()).divideScalar(positions.length);
  positions.forEach((p) => p.sub(center));
  const radius = Math.max(...positions.map((p) => p.length()), 1);

  const sphere = new THREE.SphereGeometry(1, 32, 16);
  const material = (symbol, options = {}) =>
    new THREE.MeshStandardMaterial({ color: COLORS[symbol] ?? 0xff1493, roughness: 0.5, ...options });

  // Atoms and bonds
  positions.forEach((p, i) => {
    const atom = new THREE.Mesh(sphere, material(symbols[i]));
    const size = 0.25 * (covalentRadii[symbols[i]] ?? 0.7) + 0.1;
    atom.scale.setScalar(size);
    atom.position.copy(p);
    group.add(atom);
    labels.push(atomLabel(`${symbols[i]}${i + 1}`, p, size));
  });
  const cylinder = new THREE.CylinderGeometry(0.08, 0.08, 1, 12);
  const bondMaterial = new THREE.MeshStandardMaterial({ color: 0x999999 });
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      const cutoff = BOND_FACTOR * ((covalentRadii[symbols[i]] ?? 0.7) + (covalentRadii[symbols[j]] ?? 0.7));
      const d = positions[i].distanceTo(positions[j]);
      if (d > cutoff) continue;
      const bond = new THREE.Mesh(cylinder, bondMaterial);
      bond.scale.set(1, d, 1);
      bond.position.copy(positions[i]).add(positions[j]).multiplyScalar(0.5);
      bond.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        positions[j].clone().sub(positions[i]).normalize(),
      );
      group.add(bond);
    }
  }

  // Ellipsoids: a unit sphere stretched along the principal axes
  tensors?.forEach((tensor, i) => {
    const axes = ellipsoidAxes(tensor);
    const [a, b, c] = axes.map((axis) => axis.direction);
    // A rotation (right-handed basis)
    if (a.clone().cross(b).dot(c) < 0) c.negate();
    const ellipsoid = new THREE.Mesh(
      sphere,
      material(symbols[i], { transparent: true, opacity: 0.35, depthWrite: false }),
    );
    ellipsoid.matrixAutoUpdate = false;
    ellipsoid.matrix
      .makeBasis(a, b, c)
      .multiply(new THREE.Matrix4().makeScale(...axes.map((axis) => scale * axis.length)))
      .setPosition(positions[i]);
    ellipsoids.push(ellipsoid);

    // Δ models: the principal axes, red if positive, black if negative
    if (showAxes) {
      axes.forEach((axis) => {
        const half = axis.direction.clone().multiplyScalar(0.65 * scale * axis.length);
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([positions[i].clone().sub(half), positions[i].clone().add(half)]),
          new THREE.LineBasicMaterial({ color: axis.positive ? 0xcc0000 : 0x000000 }),
        );
        ellipsoids.push(line);
      });
    }
  });

  group.add(...labels, ...ellipsoids);
  return { group, radius, labels, ellipsoids };
}

/**
 * The molecule `structure` (its atoms only) with the ellipsoids of the
 * per-atom `tensors` (row-major 3x3), scaled by `scale`; `showAxes` also draws
 * their principal axes coloured by sign. `showEllipsoids` and `showLabels`
 * switch the ellipsoids and the atom labels on and off, keeping the view.
 */
export default function MoleculeViewer({
  structure,
  tensors,
  scale = 0.25,
  showAxes = false,
  showEllipsoids = true,
  showLabels = false,
  className = "",
}) {
  const container = useRef(null);
  const layers = useRef(null);
  const visibility = useRef({});
  visibility.current = { showEllipsoids, showLabels };

  // Visibility of each object: the label renderer does not hide the children
  // of a hidden group
  const applyVisibility = () => {
    if (!layers.current) return;
    layers.current.labels.forEach((label) => (label.visible = visibility.current.showLabels));
    layers.current.ellipsoids.forEach((object) => (object.visible = visibility.current.showEllipsoids));
  };
  useEffect(applyVisibility, [showEllipsoids, showLabels]);

  useEffect(() => {
    const element = container.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    element.appendChild(renderer.domElement);
    // Labels: HTML over the canvas, not catching the mouse
    const labelRenderer = new CSS2DRenderer();
    Object.assign(labelRenderer.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
    element.appendChild(labelRenderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
    const light = new THREE.DirectionalLight(0xffffff, 1.8);
    camera.add(light);
    scene.add(camera);

    const { group, radius, labels, ellipsoids } = buildMolecule(structure, tensors, scale, showAxes);
    scene.add(group);
    layers.current = { labels, ellipsoids };
    applyVisibility();
    camera.position.set(0, 0, 3.2 * radius + 4);

    // Trackball: rotates freely, also over the poles (orbit controls keep "up")
    const controls = new TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 3;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.6;

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = element;
      renderer.setSize(w, h);
      labelRenderer.setSize(w, h);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
      controls.handleResize();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    resize();

    let frame;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      scene.traverse((object) => {
        object.geometry?.dispose();
        object.material?.dispose();
      });
      renderer.dispose();
      element.removeChild(renderer.domElement);
      element.removeChild(labelRenderer.domElement);
      layers.current = null;
    };
    // The visibility is applied by applyVisibility, without a rebuild
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structure, tensors, scale, showAxes]);

  return <div ref={container} className={`relative h-full w-full overflow-hidden ${className}`} />;
}
