import React from 'react';
import { NeuroState, LogEntry, PipelineStage, PipelineStepResult, BrainRegionType } from '../types';
import { COLORS } from '../constants';
import { 
  Activity, 
  Brain, 
  Zap, 
  ShieldCheck, 
  AlertTriangle,
  Terminal, 
  Cpu,
  Play,
  Square,
  Loader2
} from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// =============================================================================
// NEUROTRANSMITTER GAUGE
// =============================================================================
const NeuroGauge: React.FC<{ 
  label: string; 
  value: number; 
  color: string; 
  icon: React.ElementType 
}> = ({ label, value, color, icon: Icon }) => (
  <div className="flex flex-col gap-1 mb-3">
    <div className="flex justify-between items-center text-xs uppercase tracking-wider font-bold text-gray-400">
      <span className="flex items-center gap-2">
        <Icon size={14} color={color} />
        {label}
      </span>
      <span style={{ color }}>{value.toFixed(1)}%</span>
    </div>
    <div className="h-2 w-full bg-gray-900 rounded-full overflow-hidden border border-gray-800">
      <div 
        className="h-full transition-all duration-500 ease-out"
        style={{ width: `${value}%`, backgroundColor: color }}
      />
    </div>
  </div>
);

// =============================================================================
// PIPELINE STAGE INDICATOR
// =============================================================================
const StageIndicator: React.FC<{
  currentStage: PipelineStage;
  isRunning: boolean;
}> = ({ currentStage, isRunning }) => {
  const stages = [
    { key: PipelineStage.SENSORY, label: 'INPUT', color: COLORS.THALAMUS },
    { key: PipelineStage.WORKING_MEMORY, label: 'MEMORY', color: COLORS.DLPFC },
    { key: PipelineStage.METACOGNITION, label: 'META', color: COLORS.APFC },
    { key: PipelineStage.PLANNING, label: 'PLAN', color: COLORS.PFC },
    { key: PipelineStage.INHIBITION, label: 'GATE', color: COLORS.BASAL_GANGLIA },
    { key: PipelineStage.EXECUTION, label: 'EXEC', color: COLORS.MOTOR_CORTEX },
    { key: PipelineStage.REWARD, label: 'LEARN', color: COLORS.HIPPOCAMPUS },
  ];

  return (
    <div className="flex gap-1 items-center">
      {stages.map((stage, i) => {
        const isActive = currentStage === stage.key;
        const isPast = stages.findIndex(s => s.key === currentStage) > i;
        
        return (
          <React.Fragment key={stage.key}>
            <div 
              className={`
                px-2 py-1 rounded text-[9px] font-mono font-bold uppercase
                transition-all duration-300
                ${isActive 
                  ? 'bg-white/20 text-white scale-110 ring-2 ring-white/50' 
                  : isPast 
                    ? 'bg-green-500/20 text-green-400' 
                    : 'bg-gray-800/50 text-gray-600'}
              `}
              style={{ borderColor: isActive ? stage.color : 'transparent' }}
            >
              {isActive && isRunning && <Loader2 size={8} className="inline mr-1 animate-spin" />}
              {stage.label}
            </div>
            {i < stages.length - 1 && (
              <div className={`w-2 h-px ${isPast ? 'bg-green-500' : 'bg-gray-700'}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};

// =============================================================================
// ACTIVITY LOG
// =============================================================================
const ActivityLog: React.FC<{ logs: LogEntry[] }> = ({ logs }) => {
  const bottomRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="h-48 overflow-y-auto font-mono text-xs space-y-1 p-2 scrollbar-thin">
      {logs.map((log) => (
        <div key={log.id} className="flex gap-2 opacity-80 hover:opacity-100">
          <span className="text-gray-600 shrink-0">
            [{new Date(log.timestamp).toLocaleTimeString([], { hour12: false })}]
          </span>
          <span className={`
            ${log.type === 'success' ? 'text-green-400' : ''}
            ${log.type === 'error' ? 'text-red-400' : ''}
            ${log.type === 'warning' ? 'text-yellow-400' : ''}
            ${log.type === 'info' ? 'text-blue-300' : ''}
          `}>
            {log.message}
          </span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
};

// =============================================================================
// HISTORY CHART
// =============================================================================
interface HistoryPoint {
  time: string;
  dopamine: number;
  serotonin: number;
  norepinephrine: number;
}

const NeuroHistory: React.FC<{ history: HistoryPoint[] }> = ({ history }) => (
  <div className="h-24 w-full mt-2">
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={history}>
        <defs>
          <linearGradient id="colorDopa" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.PFC} stopOpacity={0.3}/>
            <stop offset="95%" stopColor={COLORS.PFC} stopOpacity={0}/>
          </linearGradient>
          <linearGradient id="colorSero" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.BASAL_GANGLIA} stopOpacity={0.3}/>
            <stop offset="95%" stopColor={COLORS.BASAL_GANGLIA} stopOpacity={0}/>
          </linearGradient>
          <linearGradient id="colorNore" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.AMYGDALA} stopOpacity={0.3}/>
            <stop offset="95%" stopColor={COLORS.AMYGDALA} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <XAxis dataKey="time" hide />
        <YAxis domain={[0, 100]} hide />
        <Tooltip 
          contentStyle={{ backgroundColor: '#111', border: '1px solid #333', fontSize: '11px' }}
        />
        <Area type="monotone" dataKey="dopamine" stroke={COLORS.PFC} fillOpacity={1} fill="url(#colorDopa)" strokeWidth={2} isAnimationActive={false} />
        <Area type="monotone" dataKey="serotonin" stroke={COLORS.BASAL_GANGLIA} fillOpacity={1} fill="url(#colorSero)" strokeWidth={2} isAnimationActive={false} />
        <Area type="monotone" dataKey="norepinephrine" stroke={COLORS.AMYGDALA} fillOpacity={1} fill="url(#colorNore)" strokeWidth={2} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  </div>
);

// =============================================================================
// MAIN DASHBOARD
// =============================================================================
interface DashboardProps {
  neuroState: NeuroState;
  currentStage: PipelineStage;
  currentStep: PipelineStepResult | null;
  logs: LogEntry[];
  neuroHistory: HistoryPoint[];
  isRunning: boolean;
  onRunPipeline: (key: string) => void;
  onStopPipeline: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ 
  neuroState, 
  currentStage,
  currentStep,
  logs, 
  neuroHistory,
  isRunning,
  onRunPipeline,
  onStopPipeline
}) => {
  return (
    <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-4 z-10">
      {/* ===== HEADER ===== */}
      <header className="flex justify-between items-start pointer-events-auto">
        <div>
          <h1 className="text-2xl font-black tracking-tighter text-white flex items-center gap-2">
            <Brain className="text-cyan-400" size={28} />
            NEURO<span className="text-cyan-400">MIMETIC</span> CORE
          </h1>
          <p className="text-gray-500 text-xs mt-1">
            PFC Agent Pipeline Visualization • v2.0
          </p>
        </div>
        
        {/* Status + Stage Indicator */}
        <div className="flex flex-col items-end gap-2">
          <div className="bg-black/80 backdrop-blur border border-gray-800 px-3 py-1.5 rounded flex items-center gap-2">
            <span className="text-xs text-gray-500 uppercase">Status</span>
            <span className={`font-mono font-bold text-sm flex items-center gap-1 ${isRunning ? 'text-cyan-400' : 'text-green-400'}`}>
              {isRunning ? 'PROCESSING' : 'IDLE'}
              <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-cyan-400 animate-pulse' : 'bg-green-500'}`} />
            </span>
          </div>
          <StageIndicator currentStage={currentStage} isRunning={isRunning} />
        </div>
      </header>

      {/* ===== MAIN CONTENT ===== */}
      <div className="flex flex-1 gap-4 items-stretch mt-4">
        
        {/* LEFT PANEL: Neurochemistry */}
        <div className="w-72 bg-black/70 backdrop-blur-md border border-gray-800/50 p-4 rounded-xl pointer-events-auto self-start">
          <h2 className="text-xs font-bold text-gray-400 mb-3 border-b border-gray-800 pb-2 flex items-center gap-2">
            <Activity size={14} /> NEUROCHEMISTRY
          </h2>
          
          <NeuroGauge label="Dopamine" value={neuroState.dopamine} color={COLORS.PFC} icon={Zap} />
          <NeuroGauge label="Serotonin" value={neuroState.serotonin} color={COLORS.BASAL_GANGLIA} icon={ShieldCheck} />
          <NeuroGauge label="Norepinephrine" value={neuroState.norepinephrine} color={COLORS.AMYGDALA} icon={AlertTriangle} />

          <NeuroHistory history={neuroHistory} />
        </div>

        {/* CENTER: Spacer for Brain */}
        <div className="flex-1" />

        {/* RIGHT PANEL: Controls + Logs */}
        <div className="w-80 flex flex-col gap-3 self-start">
          
          {/* Control Deck */}
          <div className="bg-black/70 backdrop-blur-md border border-gray-800/50 p-4 rounded-xl pointer-events-auto">
            <h2 className="text-xs font-bold text-gray-400 mb-3 border-b border-gray-800 pb-2 flex items-center gap-2">
              <Cpu size={14} /> SIMULATION PROTOCOLS
            </h2>
            
            <div className="grid grid-cols-2 gap-2">
              <button 
                onClick={() => onRunPipeline('process_input')}
                disabled={isRunning}
                className="bg-gray-800 hover:bg-cyan-900/50 disabled:opacity-50 disabled:cursor-not-allowed
                  text-xs py-2 px-3 rounded border border-gray-700 hover:border-cyan-500/50 
                  transition-all text-left group"
              >
                <div className="text-cyan-400 font-bold flex items-center gap-1">
                  <Play size={12} /> RUN_PIPELINE
                </div>
                <div className="text-gray-500 text-[10px]">Full PFC Sequence</div>
              </button>
              
              <button 
                onClick={() => onRunPipeline('threat_detected')}
                disabled={isRunning}
                className="bg-gray-800 hover:bg-red-900/50 disabled:opacity-50 disabled:cursor-not-allowed
                  text-xs py-2 px-3 rounded border border-gray-700 hover:border-red-500/50 
                  transition-all text-left group"
              >
                <div className="text-red-400 font-bold flex items-center gap-1">
                  <AlertTriangle size={12} /> THREAT_TEST
                </div>
                <div className="text-gray-500 text-[10px]">Amygdala Override</div>
              </button>
            </div>

            {isRunning && (
              <button 
                onClick={onStopPipeline}
                className="w-full mt-2 bg-red-900/30 hover:bg-red-900/50 text-red-400 
                  text-xs py-2 rounded border border-red-500/30 hover:border-red-500/50 
                  transition-all flex items-center justify-center gap-1"
              >
                <Square size={12} /> STOP
              </button>
            )}
          </div>

          {/* Console Log */}
          <div className="bg-black/80 backdrop-blur-md border border-gray-800/50 p-4 rounded-xl pointer-events-auto flex-1">
            <h2 className="text-xs font-bold text-gray-400 mb-2 border-b border-gray-800 pb-2 flex items-center gap-2">
              <Terminal size={14} /> SYSTEM LOGS
            </h2>
            <ActivityLog logs={logs} />
          </div>
        </div>
      </div>
      
      {/* ===== FOOTER ===== */}
      <div className="text-center text-xs text-gray-600 mt-2">
        Drag to rotate • Scroll to zoom • Click RUN_PIPELINE to start
      </div>
    </div>
  );
};