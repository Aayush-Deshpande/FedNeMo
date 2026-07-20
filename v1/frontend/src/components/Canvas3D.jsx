import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars } from '@react-three/drei';
import * as THREE from 'three';

/* ─────────────────────────────────────────
   ACCRETION DISK (SWIRLING GREEN/GOLD GLOW)
───────────────────────────────────────── */
function AccretionDisk() {
  const diskRef = useRef();

  const shaderMat = useMemo(() => {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vPos;
        void main() {
          vUv = uv;
          vPos = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uTime;
        varying vec2 vUv;
        varying vec3 vPos;

        void main() {
          float dist = length(vUv - vec2(0.5));
          if (dist > 0.5 || dist < 0.12) discard;

          float angle = atan(vUv.y - 0.5, vUv.x - 0.5);
          float spiral = sin(angle * 6.0 - uTime * 3.0 + dist * 25.0);
          
          float band = smoothstep(0.5, 0.2, dist) * smoothstep(0.12, 0.22, dist);
          float intensity = band * (0.6 + 0.4 * spiral);

          // Color transition: NVIDIA Green core -> Warm Amber outer edge
          vec3 green = vec3(0.46, 0.72, 0.0);
          vec3 gold = vec3(0.9, 0.5, 0.05);
          vec3 color = mix(green, gold, smoothstep(0.2, 0.45, dist));

          gl_FragColor = vec4(color * 1.5, intensity * 0.35);
        }
      `,
    });
  }, []);

  useFrame((state, delta) => {
    if (diskRef.current) {
      diskRef.current.rotation.z += delta * 0.2;
      shaderMat.uniforms.uTime.value = state.clock.elapsedTime;
    }
  });

  return (
    <mesh ref={diskRef} rotation={[-Math.PI / 2.8, 0.2, 0]} position={[2, -0.5, -2]}>
      <ringGeometry args={[1.2, 4.5, 64]} />
      <primitive object={shaderMat} attach="material" />
    </mesh>
  );
}

/* ─────────────────────────────────────────
   BLACK HOLE EVENT HORIZON & PHOTON SPHERE
───────────────────────────────────────── */
function BlackHole() {
  const holeRef = useRef();

  useFrame((state) => {
    if (holeRef.current) {
      const t = state.clock.elapsedTime;
      holeRef.current.position.y = -0.5 + Math.sin(t * 0.5) * 0.1;
    }
  });

  return (
    <group ref={holeRef} position={[2, -0.5, -2]}>
      {/* Event Horizon (Pure Black Sphere) */}
      <mesh>
        <sphereGeometry args={[1.15, 64, 64]} />
        <meshBasicMaterial color="#000000" />
      </mesh>

      {/* Photon Ring Glow */}
      <mesh>
        <sphereGeometry args={[1.22, 64, 64]} />
        <meshBasicMaterial
          color="#76b900"
          transparent
          opacity={0.25}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

/* ─────────────────────────────────────────
   SWIRLING COSMIC PARTICLES (FALLING IN)
───────────────────────────────────────── */
function CosmicVortex() {
  const pointsRef = useRef();
  const count = 1500;

  const [positions, initialData] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const data = [];
    for (let i = 0; i < count; i++) {
      const radius = 2.0 + Math.random() * 12.0;
      const angle = Math.random() * Math.PI * 2;
      const height = (Math.random() - 0.5) * 3.0;
      const speed = 0.2 + Math.random() * 0.5;
      
      pos[i * 3] = Math.cos(angle) * radius + 2;
      pos[i * 3 + 1] = height - 0.5;
      pos[i * 3 + 2] = Math.sin(angle) * radius - 2;

      data.push({ radius, angle, height, speed });
    }
    return [pos, data];
  }, []);

  useFrame((state, delta) => {
    if (!pointsRef.current) return;
    const pos = pointsRef.current.geometry.attributes.position.array;

    for (let i = 0; i < count; i++) {
      const d = initialData[i];
      d.angle += (d.speed / d.radius) * delta * 0.8;
      
      // Spiral inward slowly
      d.radius -= delta * 0.15;
      if (d.radius < 1.2) {
        d.radius = 10.0 + Math.random() * 4.0;
      }

      pos[i * 3] = Math.cos(d.angle) * d.radius + 2;
      pos[i * 3 + 1] = d.height + Math.sin(d.angle * 2.0) * 0.2 - 0.5;
      pos[i * 3 + 2] = Math.sin(d.angle) * d.radius - 2;
    }

    pointsRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={positions}
          count={count}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color="#76b900"
        transparent
        opacity={0.35}
        blending={THREE.AdditiveBlending}
        sizeAttenuation
      />
    </points>
  );
}

/* ─────────────────────────────────────────
   CANVAS CONTAINER
───────────────────────────────────────── */
export default function Canvas3D() {
  return (
    <div className="canvas-bg">
      <Canvas camera={{ position: [0, 0, 9], fov: 50 }}>
        <color attach="background" args={['#020304']} />
        
        {/* Deep space starfield */}
        <Stars radius={100} depth={50} count={3500} factor={4} saturation={0.5} fade speed={1} />
        
        {/* 3D Black Hole & Accretion Disk */}
        <BlackHole />
        <AccretionDisk />
        <CosmicVortex />
      </Canvas>
    </div>
  );
}
