import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Trail, Html, Line, Sphere } from '@react-three/drei';
import * as THREE from 'three';
import { BrainRegionType, NeuroState, PipelineStepResult, PipelineStage } from '../types';
import { BRAIN_NODES, NEURAL_PATHWAYS, COLORS } from '../constants';

// =============================================================================
// BRAIN REGION NODE - Glowing sphere with label
// =============================================================================
const BrainRegionNode: React.FC<{
  position: [number, number, number];
  color: string;
  label: string;
  isActive: boolean;
  scale?: number;
}> = ({ position, color, label, isActive, scale = 1 }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (!meshRef.current || !glowRef.current) return;
    const time = state.clock.getElapsedTime();
    
    // Pulse when active
    if (isActive) {
      const pulse = 1 + Math.sin(time * 8) * 0.2;
      meshRef.current.scale.setScalar(pulse * scale);
      glowRef.current.scale.setScalar(pulse * 2.5 * scale);
      (glowRef.current.material as THREE.MeshBasicMaterial).opacity = 0.3 + Math.sin(time * 8) * 0.15;
    } else {
      meshRef.current.scale.setScalar(scale);
      glowRef.current.scale.setScalar(scale * 1.5);
      (glowRef.current.material as THREE.MeshBasicMaterial).opacity = 0.05;
    }
  });

  return (
    <group position={position}>
      {/* Core */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.2, 32, 32]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color} 
          emissiveIntensity={isActive ? 2 : 0.3}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>
      
      {/* Glow */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshBasicMaterial 
          color={color} 
          transparent 
          opacity={0.1} 
          depthWrite={false}
        />
      </mesh>
      
      {/* Label */}
      <Html position={[0, 0.5, 0]} center distanceFactor={8}>
        <div className={`
          text-[10px] font-mono font-bold px-2 py-1 rounded whitespace-nowrap
          transition-all duration-300 select-none pointer-events-none
          ${isActive 
            ? 'bg-black/80 text-white border border-white/50 scale-110' 
            : 'bg-black/40 text-gray-400 border border-transparent'}
        `}>
          {label}
        </div>
      </Html>
    </group>
  );
};

// =============================================================================
// THOUGHT PULSE - Animated data packet traveling through the brain
// =============================================================================
const ThoughtPulse: React.FC<{
  targetRegion: BrainRegionType | null;
  color: string;
}> = ({ targetRegion, color }) => {
  const pulseRef = useRef<THREE.Mesh>(null);
  const [currentPos] = useState(() => new THREE.Vector3(0, -2, 0));
  const [targetPos, setTargetPos] = useState(() => new THREE.Vector3(0, -2, 0));
  
  // Update target when region changes
  useEffect(() => {
    if (!targetRegion) {
      setTargetPos(new THREE.Vector3(0, -2, 0));
      return;
    }
    
    const node = BRAIN_NODES.find(n => n.type === targetRegion);
    if (node) {
      setTargetPos(new THREE.Vector3(...node.position));
    }
  }, [targetRegion]);

  useFrame((state, delta) => {
    if (!pulseRef.current) return;
    
    // Smooth interpolation to target
    currentPos.lerp(targetPos, delta * 3);
    pulseRef.current.position.copy(currentPos);
    
    // Rotate for visual effect
    pulseRef.current.rotation.x += delta * 2;
    pulseRef.current.rotation.y += delta * 3;
  });

  if (!targetRegion) return null;

  return (
    <group>
      <Trail
        width={0.8}
        length={6}
        color={new THREE.Color(color)}
        attenuation={(t) => t * t}
      >
        <mesh ref={pulseRef} position={[0, -2, 0]}>
          <icosahedronGeometry args={[0.12, 0]} />
          <meshBasicMaterial color={color} toneMapped={false} />
        </mesh>
      </Trail>
      
      {/* Light following the pulse */}
      <pointLight 
        position={currentPos.toArray()} 
        intensity={3} 
        distance={3} 
        color={color} 
      />
    </group>
  );
};

// =============================================================================
// NEURAL PATHWAY - Animated connection line
// =============================================================================
const NeuralPathway: React.FC<{
  start: [number, number, number];
  end: [number, number, number];
  color: string;
  isActive: boolean;
}> = ({ start, end, color, isActive }) => {
  const points = useMemo(() => {
    // Create curved path with control point
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const mid = new THREE.Vector3().lerpVectors(startVec, endVec, 0.5);
    mid.y += 0.3; // Arc upward
    
    const curve = new THREE.QuadraticBezierCurve3(startVec, mid, endVec);
    return curve.getPoints(20);
  }, [start, end]);

  return (
    <Line
      points={points}
      color={color}
      lineWidth={isActive ? 3 : 1}
      transparent
      opacity={isActive ? 0.8 : 0.15}
    />
  );
};

// =============================================================================
// INFO CARD - Floating JSON display
// =============================================================================
const InfoCard: React.FC<{
  stepData: PipelineStepResult | null;
  position: [number, number, number];
}> = ({ stepData, position }) => {
  if (!stepData) return null;

  return (
    <Html position={position} center distanceFactor={6}>
      <div className="bg-black/90 border border-cyan-500/50 rounded-lg p-3 min-w-[280px] max-w-[320px] backdrop-blur-sm animate-fadeIn">
        {/* Header */}
        <div className="flex items-center gap-2 border-b border-cyan-500/30 pb-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-cyan-400 font-mono font-bold text-sm">
            {stepData.title}
          </span>
        </div>
        
        {/* Content */}
        <div className="space-y-2 text-xs font-mono">
          {stepData.data.input && (
            <div>
              <span className="text-gray-500">INPUT: </span>
              <span className="text-yellow-300">{stepData.data.input}</span>
            </div>
          )}
          
          {stepData.data.processing && (
            <div>
              <span className="text-gray-500">PROC: </span>
              <span className="text-blue-300">{stepData.data.processing}</span>
            </div>
          )}
          
          {stepData.data.output && (
            <div>
              <span className="text-gray-500">OUT: </span>
              <span className="text-green-300">{stepData.data.output}</span>
            </div>
          )}
          
          {stepData.data.decision && (
            <div className={`font-bold ${stepData.data.decision === 'APPROVED' ? 'text-green-400' : 'text-red-400'}`}>
              DECISION: {stepData.data.decision}
            </div>
          )}
        </div>
      </div>
    </Html>
  );
};

// =============================================================================
// MAIN BRAIN SCENE
// =============================================================================
interface BrainSceneProps {
  activeRegion: BrainRegionType | null;
  currentStep: PipelineStepResult | null;
  neuroState: NeuroState;
}

export const BrainScene: React.FC<BrainSceneProps> = ({ 
  activeRegion, 
  currentStep,
  neuroState 
}) => {
  // Calculate mood color
  const moodColor = useMemo(() => {
    const { dopamine, serotonin, norepinephrine } = neuroState;
    if (dopamine > serotonin && dopamine > norepinephrine) return COLORS.PFC;
    if (serotonin > dopamine && serotonin > norepinephrine) return COLORS.BASAL_GANGLIA;
    if (norepinephrine > dopamine && norepinephrine > serotonin) return COLORS.AMYGDALA;
    return '#ffffff';
  }, [neuroState]);

  // Get active node position for info card
  const activeNodePosition = useMemo(() => {
    if (!activeRegion) return [3, 1, 0] as [number, number, number];
    const node = BRAIN_NODES.find(n => n.type === activeRegion);
    if (node) {
      return [node.position[0] + 2, node.position[1] + 0.5, node.position[2]] as [number, number, number];
    }
    return [3, 1, 0] as [number, number, number];
  }, [activeRegion]);

  return (
    <div className="w-full h-full absolute inset-0">
      <Canvas camera={{ position: [7, 3, 7], fov: 40 }} gl={{ antialias: true }}>
        <color attach="background" args={['#030308']} />
        
        {/* Lighting */}
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1.5} color="#ffffff" />
        <pointLight position={[-10, -5, -10]} intensity={2} color={moodColor} />
        
        {/* Brain Structure */}
        <group rotation={[0, -Math.PI / 6, 0]}>
          {/* Render all region nodes */}
          {BRAIN_NODES.map((node) => (
            <BrainRegionNode
              key={node.id}
              position={node.position}
              color={node.color}
              label={node.label}
              isActive={activeRegion === node.type}
              scale={node.type === BrainRegionType.PFC ? 1.3 : 1}
            />
          ))}
          
          {/* Render neural pathways */}
          {NEURAL_PATHWAYS.map((path, i) => {
            const fromNode = BRAIN_NODES.find(n => n.id === path.from);
            const toNode = BRAIN_NODES.find(n => n.id === path.to);
            if (!fromNode || !toNode) return null;
            
            const isActive = activeRegion === fromNode.type || activeRegion === toNode.type;
            
            return (
              <NeuralPathway
                key={i}
                start={fromNode.position}
                end={toNode.position}
                color={fromNode.color}
                isActive={isActive}
              />
            );
          })}
          
          {/* Thought Pulse */}
          <ThoughtPulse 
            targetRegion={activeRegion} 
            color={currentStep ? COLORS[activeRegion?.replace(/ /g, '_').toUpperCase() || 'PFC'] || '#00f3ff' : '#00f3ff'}
          />
          
          {/* Info Card */}
          <InfoCard stepData={currentStep} position={activeNodePosition} />
        </group>
        
        <OrbitControls 
          enablePan={false} 
          minDistance={6} 
          maxDistance={15} 
          autoRotate={!activeRegion}
          autoRotateSpeed={0.3}
        />
      </Canvas>
    </div>
  );
};
