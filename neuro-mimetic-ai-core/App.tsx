import React, { useState, useCallback, useRef } from 'react';
import { BrainScene } from './components/BrainScene';
import { Dashboard } from './components/Dashboard';
import { 
  BrainRegionType, 
  NeuroState, 
  LogEntry, 
  PipelineStepResult,
  PipelineStage 
} from './types';
import { PIPELINE_SEQUENCES } from './constants';

// =============================================================================
// INITIAL STATE
// =============================================================================
const INITIAL_NEURO: NeuroState = {
  dopamine: 50,
  serotonin: 70,
  norepinephrine: 40,
};

// =============================================================================
// MAIN APP COMPONENT
// =============================================================================
const App: React.FC = () => {
  // --- State ---
  const [neuroState, setNeuroState] = useState<NeuroState>(INITIAL_NEURO);
  const [activeRegion, setActiveRegion] = useState<BrainRegionType | null>(null);
  const [currentStep, setCurrentStep] = useState<PipelineStepResult | null>(null);
  const [currentStage, setCurrentStage] = useState<PipelineStage>(PipelineStage.IDLE);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [history, setHistory] = useState<Array<{time: string} & NeuroState>>([]);
  
  // Ref for cancellation
  const cancelRef = useRef(false);

  // --- Helpers ---
  const addLog = useCallback((
    message: string, 
    type: LogEntry['type'] = 'info', 
    stage?: PipelineStage,
    region?: BrainRegionType
  ) => {
    setLogs(prev => [...prev, {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: Date.now(),
      message,
      type,
      stage,
      region
    }].slice(-100));
  }, []);

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  // --- Pipeline Runner ---
  const runPipeline = useCallback(async (pipelineKey: string) => {
    const pipeline = PIPELINE_SEQUENCES[pipelineKey];
    if (!pipeline || isRunning) return;

    setIsRunning(true);
    cancelRef.current = false;
    addLog(`Starting pipeline: ${pipelineKey.toUpperCase()}`, 'info');

    for (const step of pipeline) {
      if (cancelRef.current) {
        addLog('Pipeline cancelled by user', 'warning');
        break;
      }

      // Update state
      setCurrentStage(step.stage);
      setActiveRegion(step.region);
      setCurrentStep(step);
      
      addLog(
        `[${step.stage}] ${step.title}`, 
        step.data.decision === 'BLOCKED' ? 'error' : 'info',
        step.stage,
        step.region
      );

      // Apply chemical changes if specified
      if (step.data.chemicals) {
        setNeuroState(prev => ({
          dopamine: step.data.chemicals?.dopamine ?? prev.dopamine,
          serotonin: step.data.chemicals?.serotonin ?? prev.serotonin,
          norepinephrine: step.data.chemicals?.norepinephrine ?? prev.norepinephrine,
        }));
      }

      // Wait for step duration
      await sleep(step.duration);
    }

    // Reset
    setIsRunning(false);
    setActiveRegion(null);
    setCurrentStep(null);
    setCurrentStage(PipelineStage.IDLE);
    addLog('Pipeline complete', 'success');
  }, [isRunning, addLog]);

  // --- Stop Pipeline ---
  const stopPipeline = useCallback(() => {
    cancelRef.current = true;
    setIsRunning(false);
    setActiveRegion(null);
    setCurrentStep(null);
    setCurrentStage(PipelineStage.IDLE);
  }, []);

  // --- Update History (for chart) ---
  React.useEffect(() => {
    const interval = setInterval(() => {
      setHistory(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        ...neuroState
      }].slice(-30));
    }, 1000);
    return () => clearInterval(interval);
  }, [neuroState]);

  // ==========================================================================
  // RENDER
  // ==========================================================================
  return (
    <div className="w-full h-screen relative bg-black overflow-hidden">
      {/* 3D Brain Visualization */}
      <BrainScene 
        activeRegion={activeRegion}
        currentStep={currentStep}
        neuroState={neuroState}
      />
      
      {/* Dashboard Overlay */}
      <Dashboard 
        neuroState={neuroState}
        currentStage={currentStage}
        currentStep={currentStep}
        logs={logs}
        neuroHistory={history}
        isRunning={isRunning}
        onRunPipeline={runPipeline}
        onStopPipeline={stopPipeline}
      />
    </div>
  );
};

export default App;