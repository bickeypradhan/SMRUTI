
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type, Modality, LiveServerMessage, Blob } from "@google/genai";
import ReactMarkdown from 'react-markdown';
import { 
  Plus, 
  History, 
  BrainCircuit, 
  LayoutDashboard, 
  ChevronRight, 
  Save, 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX,
  ArrowLeft,
  Clock,
  CheckCircle2,
  AlertCircle,
  Archive,
  Trash2,
  X,
  Menu,
  Sun,
  Moon,
  Tag,
  BarChart3,
  RefreshCw,
  ExternalLink,
  Globe,
  SendHorizontal,
  Sparkles,
  Zap,
  PhoneOff,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen
} from 'lucide-react';

// --- Core Utility Functions for Audio (Required for Live API) ---

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createAudioBlob(data: Float32Array): Blob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// --- Types & Interfaces ---

type DecisionStatus = 'active' | 'archived';

interface Decision {
  id: string;
  rootId: string;
  title: string;
  category: string;
  intent: string;
  constraints: string[];
  alternatives: string[];
  rejectedReasons: string[];
  finalDecision: string;
  confidence: string;
  reasoning: string;
  version: number;
  status: DecisionStatus;
  createdAt: number;
  updatedAt: number;
  updatedBy: 'user' | 'ai';
}

interface AIResponse {
  intent: 'assist' | 'update' | 'clarify' | 'general' | 'mixed';
  explanation: string;
  proposedChange?: Partial<Decision>;
  confidence: number;
  groundingChunks?: any[];
}

interface Interaction {
  id: string;
  query: string;
  response: AIResponse;
  timestamp: number;
}

interface ChatThread {
  id: string;
  title: string;
  interactions: Interaction[];
  updatedAt: number;
  createdAt: number;
}

// --- Database Service (IndexedDB) ---

const DB_NAME = 'SmrutiDB';
const DB_VERSION = 2; // Incremented for chatThreads
const DECISION_STORE = 'decisions';
const THREAD_STORE = 'chatThreads';

const initDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(DECISION_STORE)) {
        const store = db.createObjectStore(DECISION_STORE, { keyPath: 'id' });
        store.createIndex('rootId', 'rootId', { unique: false });
        store.createIndex('status', 'status', { unique: false });
      }
      if (!db.objectStoreNames.contains(THREAD_STORE)) {
        db.createObjectStore(THREAD_STORE, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

const storage = {
  async saveDecision(decision: Decision): Promise<void> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(DECISION_STORE, 'readwrite');
      transaction.objectStore(DECISION_STORE).put(decision);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  },
  async getAllDecisions(): Promise<Decision[]> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(DECISION_STORE, 'readonly');
      const request = transaction.objectStore(DECISION_STORE).getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  },
  async saveThread(thread: ChatThread): Promise<void> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(THREAD_STORE, 'readwrite');
      transaction.objectStore(THREAD_STORE).put(thread);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  },
  async getAllThreads(): Promise<ChatThread[]> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(THREAD_STORE, 'readonly');
      const request = transaction.objectStore(THREAD_STORE).getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  },
  async deleteThread(id: string): Promise<void> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(THREAD_STORE, 'readwrite');
      transaction.objectStore(THREAD_STORE).delete(id);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  }
};

// --- AI Service ---

const getAI = () => new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

// Global audio state to manage interruptions
let activeAudioSource: AudioBufferSourceNode | null = null;
let activeAudioCtx: AudioContext | null = null;

const stopSpeaking = () => {
  if (activeAudioSource) {
    try {
      activeAudioSource.stop();
    } catch (e) {
      // Audio already stopped
    }
    activeAudioSource = null;
  }
};

const speakText = async (text: string) => {
  // Always clear existing speech before starting new logic
  stopSpeaking();
  
  const ai = getAI();
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: `Respond naturally to: ${text}` }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
        },
      },
    });
    
    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
      if (!activeAudioCtx) {
        activeAudioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      }
      
      const buffer = await decodeAudioData(decode(base64Audio), activeAudioCtx, 24000, 1);
      const source = activeAudioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(activeAudioCtx.destination);
      
      activeAudioSource = source;
      source.start();
      
      source.onended = () => {
        if (activeAudioSource === source) activeAudioSource = null;
      };
    }
  } catch (e) {
    console.error("Audio generation failed", e);
  }
};

const generateThreadTitle = async (firstQuery: string): Promise<string> => {
  const genAI = getAI();
  const response = await genAI.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Generate a very concise 2-3 word title for a chat thread starting with: "${firstQuery}". Respond ONLY with the title.`,
  });
  return response.text?.trim() || "Logic Inquiry";
};

const parseIntent = async (query: string, context: Decision[], history: Interaction[]): Promise<AIResponse> => {
  const genAI = getAI();
  const latestDecisions = context.filter(d => d.status === 'active');
  
  const contextStr = JSON.stringify(latestDecisions.map(d => ({
    title: d.title,
    finalDecision: d.finalDecision,
    reasoning: d.reasoning,
    category: d.category,
    constraints: d.constraints,
    alternatives: d.alternatives,
    rootId: d.rootId,
    version: d.version
  })));

  const chatContents = history.flatMap(h => ([
    { role: 'user', parts: [{ text: h.query }] },
    { role: 'model', parts: [{ text: JSON.stringify(h.response) }] }
  ]));
  chatContents.push({ role: 'user', parts: [{ text: query }] });

  const response = await genAI.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: chatContents,
    config: {
      tools: [{ googleSearch: {} }],
      systemInstruction: `You are SMRUTI, the Decision Memory Continuity System.
      Maintain logical continuity over multi-turn conversations.

      HIGHLIGHTING RULE:
      Wrap key decisions, primary logic points, and the single most important summary sentence in double asterisks **like this**. 
      Focus on highlighting the *why* and the *final outcome*.

      SOURCES:
      1. INTERNAL MEMORY: context data: ${contextStr}
      2. EXTERNAL KNOWLEDGE: Google Search tool.

      INTENTS:
      - "assist": Query about existing memory.
      - "update": Changing a decision.
      - "general": Non-memory general query.
      - "mixed": Memory + web search synthesis.

      Respond ONLY in JSON format. Use standard Markdown for the "explanation" field.`,
      responseMimeType: "application/json"
    }
  });

  try {
    const text = response.text || '{}';
    const result = JSON.parse(text) as AIResponse;
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    if (groundingChunks) result.groundingChunks = groundingChunks;
    return result;
  } catch (e) {
    return { intent: 'clarify', explanation: "I couldn't synthesize the logic path from the current thread. Could you try rephrasing?", confidence: 0 };
  }
};

// --- Custom Components ---

const AnimatedMarkdown = ({ content, intent }: { content: string, intent: string }) => {
  const getHighlightClass = () => {
    switch(intent) {
      case 'update': return 'highlight-amber';
      case 'general':
      case 'mixed': return 'highlight-indigo';
      default: return 'highlight-slate';
    }
  };

  return (
    <ReactMarkdown
      components={{
        strong: ({ node, ...props }) => (
          <strong className={`animated-highlight animate-highlightReveal ${getHighlightClass()}`} {...props} />
        )
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

const Button = ({ children, onClick, variant = 'primary', className = '', icon: Icon, disabled = false, type = "button" }: any) => {
  const variants = {
    primary: 'bg-slate-900 text-white hover:bg-slate-800 dark:bg-slate-50 dark:text-slate-950 dark:hover:bg-slate-200 disabled:opacity-50 active:scale-95 shadow-sm',
    secondary: 'bg-white text-slate-900 border border-slate-200 hover:bg-slate-50 dark:bg-slate-900 dark:text-slate-50 dark:border-slate-800 dark:hover:bg-slate-800 active:scale-95 shadow-sm',
    ghost: 'text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800 active:scale-95',
    danger: 'text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/30 active:scale-95'
  };
  return (
    <button type={type} disabled={disabled} onClick={onClick} className={`px-4 py-2.5 rounded-xl font-semibold transition-all flex items-center justify-center gap-2 touch-manipulation ${variants[variant as keyof typeof variants]} ${className}`}>
      {Icon && <Icon size={18} />}
      {children}
    </button>
  );
};

const Card = ({ children, className = '' }: any) => (
  <div className={`bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl overflow-hidden shadow-sm ${className}`}>
    {children}
  </div>
);

// --- Live Assistant Mode ---

const LiveAssistantOverlay = ({ onClose, decisions }: { onClose: () => void, decisions: Decision[] }) => {
  const [isConnecting, setIsConnecting] = useState(true);
  const [isActive, setIsActive] = useState(false);
  const nextStartTimeRef = useRef(0);
  const sourcesRef = useRef(new Set<AudioBufferSourceNode>());
  const sessionRef = useRef<any>(null);
  const audioContextsRef = useRef<{ input?: AudioContext; output?: AudioContext }>({});

  useEffect(() => {
    let scriptProcessor: ScriptProcessorNode;
    let stream: MediaStream;

    const startSession = async () => {
      const ai = getAI();
      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      audioContextsRef.current = { input: inputCtx, output: outputCtx };

      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        const sessionPromise = ai.live.connect({
          model: 'gemini-2.5-flash-native-audio-preview-09-2025',
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
            systemInstruction: `You are SMRUTI, the decision memory system. You communicate via voice in a natural, friendly, and helpful manner. You have access to the user's decision context: ${JSON.stringify(decisions.map(d => d.title))}. Assist the user with their memory and reasoning.`
          },
          callbacks: {
            onopen: () => {
              setIsConnecting(false);
              setIsActive(true);
              const source = inputCtx.createMediaStreamSource(stream);
              scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
              scriptProcessor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmBlob = createAudioBlob(inputData);
                sessionPromise.then(session => session.sendRealtimeInput({ media: pcmBlob }));
              };
              source.connect(scriptProcessor);
              scriptProcessor.connect(inputCtx.destination);
            },
            onmessage: async (msg: LiveServerMessage) => {
              const base64Audio = msg.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
              if (base64Audio && outputCtx) {
                nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputCtx.currentTime);
                const buffer = await decodeAudioData(decode(base64Audio), outputCtx, 24000, 1);
                const source = outputCtx.createBufferSource();
                source.buffer = buffer;
                source.connect(outputCtx.destination);
                source.start(nextStartTimeRef.current);
                nextStartTimeRef.current += buffer.duration;
                sourcesRef.current.add(source);
                source.onended = () => sourcesRef.current.delete(source);
              }
              if (msg.serverContent?.interrupted) {
                sourcesRef.current.forEach(s => { try { s.stop(); } catch(e) {} });
                sourcesRef.current.clear();
                nextStartTimeRef.current = 0;
              }
            },
            onclose: () => onClose(),
            onerror: (e) => { console.error("Live Error", e); onClose(); }
          }
        });

        sessionRef.current = await sessionPromise;
      } catch (e) {
        console.error("Failed to start live session", e);
        onClose();
      }
    };

    startSession();

    return () => {
      if (scriptProcessor) scriptProcessor.disconnect();
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (audioContextsRef.current.input) audioContextsRef.current.input.close();
      if (audioContextsRef.current.output) audioContextsRef.current.output.close();
      if (sessionRef.current) sessionRef.current.close();
    };
  }, []);

  return (
    <div className="fixed inset-0 bg-slate-900/90 backdrop-blur-3xl z-[100] flex flex-col items-center justify-center p-8 animate-fadeIn text-white">
      <div className="absolute top-8 right-8">
        <Button variant="ghost" onClick={onClose} className="text-white hover:bg-white/10 p-4 rounded-full">
          <PhoneOff size={32} />
        </Button>
      </div>

      <div className="relative flex items-center justify-center w-64 h-64 md:w-80 md:h-80">
        <div className={`absolute inset-0 rounded-full border-4 border-dashed border-white/20 ${isActive ? 'animate-[spin_20s_linear_infinite]' : ''}`}></div>
        <div className={`absolute inset-4 rounded-full border-2 border-dotted border-white/40 ${isActive ? 'animate-[spin_10s_linear_infinite_reverse]' : ''}`}></div>
        
        <div className={`relative z-10 w-32 h-32 md:w-40 md:h-40 bg-white/10 backdrop-blur-xl rounded-full flex items-center justify-center shadow-2xl ${isActive ? 'animate-pulse' : ''}`}>
          <BrainCircuit size={isActive ? 64 : 48} className={`transition-all duration-500 ${isActive ? 'text-white' : 'text-white/40'}`} />
        </div>
        
        {isActive && (
          <>
            <div className="absolute inset-[-20px] rounded-full border border-white/10 animate-[ping_3s_linear_infinite]"></div>
            <div className="absolute inset-[-40px] rounded-full border border-white/5 animate-[ping_4s_linear_infinite]"></div>
          </>
        )}
      </div>

      <div className="mt-16 text-center space-y-4">
        <h2 className="text-3xl font-black tracking-tighter uppercase">
          {isConnecting ? 'Initializing Synapse...' : 'Active'}
        </h2>
        <p className="text-slate-400 font-medium tracking-widest text-sm uppercase opacity-80">
          {isConnecting ? 'Calibrating neural frequency...' : 'Listening to your logic. Speak naturally.'}
        </p>
      </div>

      {!isConnecting && (
        <div className="absolute bottom-12 flex gap-1 items-end h-8">
          {[1,2,3,4,5,6,7,8].map(i => (
             <div key={i} className={`w-1 bg-white rounded-full transition-all duration-300 ${isActive ? 'animate-[bounce_1s_infinite]' : 'h-2'}`} style={{ animationDelay: `${i * 0.1}s`, height: isActive ? `${Math.random() * 20 + 10}px` : '4px' }}></div>
          ))}
        </div>
      )}
    </div>
  );
};

// --- Ask SMRUTI (with History) ---

const AskSmruti = ({ decisions, onUpdate }: any) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [isVoiceOutputEnabled, setIsVoiceOutputEnabled] = useState(true);
  const [isLiveOverlayOpen, setIsLiveOverlayOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    loadThreads();
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
    }
  }, [threads, activeThreadId, loading]);

  const loadThreads = async () => {
    const all = await storage.getAllThreads();
    setThreads(all.sort((a, b) => b.updatedAt - a.updatedAt));
  };

  const activeThread = useMemo(() => 
    threads.find(t => t.id === activeThreadId) || null,
  [threads, activeThreadId]);

  const toggleVoiceOutput = () => {
    const nextValue = !isVoiceOutputEnabled;
    setIsVoiceOutputEnabled(nextValue);
    // If sound is turned off, kill current speech immediately
    if (!nextValue) {
      stopSpeaking();
    }
  };

  const handleAsk = async () => {
    if (!query.trim() || loading) return;
    const currentQuery = query;
    setQuery('');
    setLoading(true);
    
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    try {
      const history = activeThread?.interactions || [];
      const res = await parseIntent(currentQuery, decisions, history);
      
      const newInteraction: Interaction = {
        id: Math.random().toString(36).substring(2),
        query: currentQuery,
        response: res,
        timestamp: Date.now()
      };

      let updatedThread: ChatThread;

      if (!activeThreadId) {
        const title = await generateThreadTitle(currentQuery);
        updatedThread = {
          id: Math.random().toString(36).substring(2, 11),
          title,
          interactions: [newInteraction],
          createdAt: Date.now(),
          updatedAt: Date.now()
        };
        setActiveThreadId(updatedThread.id);
      } else {
        updatedThread = {
          ...activeThread!,
          interactions: [...activeThread!.interactions, newInteraction],
          updatedAt: Date.now()
        };
      }

      await storage.saveThread(updatedThread);
      await loadThreads();
      
      // Auto-narrate only if enabled
      if (isVoiceOutputEnabled && res.explanation) {
        speakText(res.explanation);
      }
    } catch (e) { 
      console.error(e); 
    }
    setLoading(false);
  };

  const startNewThread = () => {
    stopSpeaking(); // Stop any speech when clearing the context
    setActiveThreadId(null);
    setQuery('');
  };

  const deleteThread = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await storage.deleteThread(id);
    if (activeThreadId === id) {
      stopSpeaking();
      setActiveThreadId(null);
    }
    await loadThreads();
  };

  const toggleMic = () => {
    if (!('webkitSpeechRecognition' in window)) return alert("Speech recognition not available.");
    const recognition = new (window as any).webkitSpeechRecognition();
    recognition.onstart = () => setIsListening(true);
    recognition.onresult = (e: any) => { setQuery(e.results[0][0].transcript); setIsListening(false); };
    recognition.onerror = () => setIsListening(false);
    recognition.start();
  };

  return (
    <div className="h-full flex relative w-full overflow-hidden bg-white dark:bg-slate-950">
      {/* Sidebar for History */}
      <aside className={`${isSidebarOpen ? 'w-72' : 'w-0'} transition-all duration-300 border-r border-slate-100 dark:border-slate-800 flex flex-col bg-slate-50/50 dark:bg-slate-900/50 overflow-hidden shrink-0`}>
        <div className="p-4 border-b border-slate-100 dark:border-slate-800">
           <Button variant="primary" onClick={startNewThread} className="w-full gap-2 text-sm" icon={Plus}>
             New Inquiry
           </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {threads.map(t => (
            <div 
              key={t.id} 
              onClick={() => {
                stopSpeaking(); // Stop speech when switching threads
                setActiveThreadId(t.id);
              }}
              className={`group flex items-center justify-between p-3 rounded-xl cursor-pointer transition-all ${activeThreadId === t.id ? 'bg-slate-200 dark:bg-slate-800 text-slate-900 dark:text-slate-50' : 'text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800/50'}`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <MessageSquare size={16} className="shrink-0 opacity-50" />
                <span className="text-sm font-semibold truncate pr-2">{t.title}</span>
              </div>
              <button 
                onClick={(e) => deleteThread(e, t.id)}
                className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-100 hover:text-red-500 rounded-lg transition-all"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
          {threads.length === 0 && (
            <div className="py-12 text-center">
               <Clock size={32} className="mx-auto text-slate-200 mb-2" />
               <p className="text-[10px] font-black uppercase text-slate-300 tracking-widest">No History</p>
            </div>
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative min-w-0 h-full">
        <header className="py-4 px-6 shrink-0 flex justify-between items-center bg-white/80 dark:bg-slate-950/80 backdrop-blur-md sticky top-0 z-20 border-b border-slate-50 dark:border-slate-800">
          <div className="flex items-center gap-4">
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 text-slate-400 hover:text-slate-900 dark:hover:text-slate-50 transition-colors">
              {isSidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}
            </button>
            <div>
              <h2 className="text-xl font-black text-slate-900 dark:text-slate-50 tracking-tighter">
                {activeThread ? activeThread.title : 'Cognitive Synthesis'}
              </h2>
              <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest font-mono">Status: Neural Link Active</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
             <Button variant="ghost" onClick={toggleVoiceOutput} className="p-2.5">
               {isVoiceOutputEnabled ? <Volume2 size={18} className="text-slate-900 dark:text-white" /> : <VolumeX size={18} className="text-slate-400" />}
             </Button>
             <Button variant="secondary" onClick={() => setIsLiveOverlayOpen(true)} className="gap-2 bg-slate-900 text-white dark:bg-white dark:text-slate-950 border-none px-4 py-1.5 text-xs" icon={Zap}>
               Go Live
             </Button>
          </div>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 pt-8 pb-56 space-y-12 no-scrollbar">
          {!activeThread && !loading && (
            <div className="h-full flex flex-col items-center justify-center text-center py-20 animate-fadeIn opacity-50">
              <Sparkles size={64} className="mb-6 text-slate-100 dark:text-slate-800" />
              <h3 className="text-xl font-bold font-serif italic mb-2 text-slate-400">Initialize logic inquiry.</h3>
              <p className="text-sm text-slate-400 max-w-xs">Ask SMRUTI to synthesize memory or search external logic.</p>
            </div>
          )}

          {activeThread?.interactions.map((interaction) => (
            <div key={interaction.id} className="animate-fadeIn space-y-6">
              <div className="flex flex-col items-end">
                <div className="bg-slate-100 dark:bg-slate-800 px-5 py-3.5 rounded-2xl rounded-tr-none max-w-[85%] shadow-sm border border-slate-200 dark:border-slate-700">
                  <p className="text-slate-900 dark:text-slate-100 font-semibold text-base md:text-lg">{interaction.query}</p>
                </div>
              </div>

              <Card className={`p-8 md:p-12 border-l-[8px] md:border-l-[16px] overflow-visible transition-all duration-700 ${interaction.response.intent === 'update' ? 'border-l-amber-500' : interaction.response.intent === 'general' ? 'border-l-indigo-500' : 'border-l-slate-900 dark:border-l-slate-50'}`}>
                <div className="mb-6 flex justify-between items-center">
                  <span className={`px-2.5 py-1 rounded-lg text-[9px] font-black uppercase tracking-[0.2em] flex items-center gap-1.5 ${interaction.response.intent === 'update' ? 'bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-400' : interaction.response.intent === 'general' ? 'bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-400' : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'}`}>
                    {interaction.response.intent}
                  </span>
                  <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest font-mono">{new Date(interaction.timestamp).toLocaleTimeString()}</span>
                </div>
                
                <div className="text-slate-800 dark:text-slate-100 text-lg md:text-xl leading-relaxed prose dark:prose-invert max-w-none font-medium">
                  <AnimatedMarkdown content={interaction.response.explanation} intent={interaction.response.intent} />
                </div>

                {interaction.response.intent === 'update' && interaction.response.proposedChange && (
                  <div className="mt-10 bg-amber-50/50 dark:bg-amber-950/20 p-6 md:p-8 rounded-3xl border border-amber-200 dark:border-amber-900 shadow-inner group">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-amber-100 dark:bg-amber-900 rounded-xl text-amber-600 dark:text-amber-400 group-hover:scale-110 transition-transform">
                        <RefreshCw size={24} className="animate-spin-slow"/>
                      </div>
                      <div>
                        <h4 className="font-black text-amber-900 dark:text-amber-100 text-lg">Evolve Path</h4>
                        <p className="text-[9px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-widest">Logic Modification</p>
                      </div>
                    </div>
                    
                    <div className="space-y-4 mb-8">
                       {interaction.response.proposedChange.finalDecision && (
                         <div className="p-4 bg-white/80 dark:bg-slate-900/80 rounded-xl border border-amber-200/50">
                            <p className="text-[8px] font-black text-amber-500 uppercase tracking-widest mb-1">New Path</p>
                            <p className="text-amber-950 dark:text-amber-50 font-bold italic">"{interaction.response.proposedChange.finalDecision}"</p>
                         </div>
                       )}
                    </div>

                    <Button onClick={() => onUpdate(interaction.response.proposedChange)} className="bg-amber-600 dark:bg-amber-500 hover:bg-amber-700 dark:hover:bg-amber-400 text-white dark:text-slate-950 border-none w-full py-4 font-black rounded-xl">Commit Memory Evolution</Button>
                  </div>
                )}
              </Card>
            </div>
          ))}

          {loading && (
            <div className="flex flex-col items-center py-12 gap-5 animate-fadeIn">
              <div className="w-10 h-10 border-[4px] border-slate-100 dark:border-slate-800 border-t-slate-900 dark:border-t-slate-50 rounded-full animate-spin"></div>
              <p className="text-[10px] font-black text-slate-400 dark:text-slate-600 uppercase tracking-[0.4em] animate-pulse">Scanning Logic Corridors...</p>
            </div>
          )}
        </div>

        {/* INPUT AREA */}
        <div className="absolute bottom-20 md:bottom-8 left-0 right-0 px-6 z-40">
          <div className="max-w-3xl mx-auto bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl border-2 border-slate-100 dark:border-slate-800 rounded-[2rem] shadow-2xl p-2 md:p-3 ring-8 ring-slate-900/[0.03] dark:ring-white/[0.03]">
            <div className="flex items-end gap-2">
              <button 
                onClick={toggleMic} 
                className={`p-3.5 rounded-full transition-all shrink-0 mb-0.5 ${isListening ? 'bg-red-500 text-white animate-pulse' : 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'}`}
              >
                <Mic size={20} strokeWidth={2.5} />
              </button>
              
              <textarea 
                ref={textareaRef}
                className="flex-1 bg-transparent px-2 md:px-4 py-3 text-base md:text-lg font-medium text-slate-900 dark:text-slate-50 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none resize-none max-h-40 overflow-y-auto no-scrollbar min-h-[44px]"
                placeholder="Engage SMRUTI..."
                rows={1}
                value={query}
                onChange={e => {
                  setQuery(e.target.value);
                  e.target.style.height = 'auto';
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
                }}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleAsk();
                  }
                }}
              />
              
              <button 
                onClick={handleAsk} 
                disabled={loading || !query.trim()} 
                className={`rounded-full p-3.5 h-[48px] w-[48px] flex items-center justify-center transition-all shadow-lg active:scale-90 disabled:opacity-30 mb-0.5 ${query.trim() ? 'bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-950' : 'bg-slate-100 dark:bg-slate-800 text-slate-400'}`}
              >
                <SendHorizontal size={22} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {isLiveOverlayOpen && <LiveAssistantOverlay onClose={() => setIsLiveOverlayOpen(false)} decisions={decisions} />}
    </div>
  );
};

// --- Theme Toggle ---

const ThemeToggle = ({ theme, toggleTheme }: { theme: string, toggleTheme: () => void }) => (
  <button onClick={toggleTheme} className="p-3 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 rounded-2xl hover:scale-110 transition-all duration-500 shadow-inner group relative overflow-hidden">
    <div className="relative z-10">{theme === 'light' ? <Moon size={20} strokeWidth={2.5} /> : <Sun size={20} strokeWidth={2.5} />}</div>
    <div className="absolute inset-0 bg-gradient-to-tr from-slate-200 to-white dark:from-slate-900 dark:to-slate-700 opacity-0 group-hover:opacity-100 transition-opacity"></div>
  </button>
);

// --- Navbar & Stats ---

const Navbar = ({ activeTab, setActiveTab, theme, toggleTheme }: { activeTab: string, setActiveTab: (t: string) => void, theme: string, toggleTheme: () => void }) => (
  <nav className="hidden md:flex w-64 bg-slate-50 dark:bg-slate-950 border-r border-slate-200 dark:border-slate-800 flex-col h-screen sticky top-0 shrink-0">
    <div className="p-8 flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-black tracking-tighter text-slate-900 dark:text-slate-50 flex items-center gap-2">
          <BrainCircuit className="text-slate-900 dark:text-slate-50" strokeWidth={3} /> SMRUTI
        </h1>
      </div>
      <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
    </div>
    <div className="flex-1 px-4 space-y-1.5">
      <NavItem icon={LayoutDashboard} label="Dashboard" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
      <NavItem icon={Plus} label="New Decision" active={activeTab === 'new'} onClick={() => setActiveTab('new')} />
      <NavItem icon={History} label="Timeline" active={activeTab === 'timeline'} onClick={() => setActiveTab('timeline')} />
      <NavItem icon={BrainCircuit} label="Inquiry" active={activeTab === 'ask'} onClick={() => setActiveTab('ask')} />
    </div>
    <div className="p-8 text-[10px] text-slate-400 dark:text-slate-600 border-t border-slate-200 dark:border-slate-800 uppercase tracking-[0.2em] font-bold">Encrypted Local Core</div>
  </nav>
);

const MobileNavbar = ({ activeTab, setActiveTab }: { activeTab: string, setActiveTab: (t: string) => void }) => (
  <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-t border-slate-200 dark:border-slate-800 flex justify-around items-center px-4 pb-safe z-50 h-20 shadow-[0_-8px_30px_rgb(0,0,0,0.04)]">
    <MobileNavItem icon={LayoutDashboard} label="Dash" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
    <MobileNavItem icon={Plus} label="Add" active={activeTab === 'new'} onClick={() => setActiveTab('new')} />
    <MobileNavItem icon={History} label="Paths" active={activeTab === 'timeline'} onClick={() => setActiveTab('timeline')} />
    <MobileNavItem icon={BrainCircuit} label="Ask" active={activeTab === 'ask'} onClick={() => setActiveTab('ask')} />
  </nav>
);

const NavItem = ({ icon: Icon, label, active, onClick }: any) => (
  <button onClick={onClick} className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all ${active ? 'bg-slate-900 text-white dark:bg-slate-50 dark:text-slate-950 shadow-lg shadow-slate-900/10 dark:shadow-white/5' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-200/50 dark:hover:bg-slate-800/50'}`}>
    <Icon size={20} strokeWidth={active ? 2.5 : 2} />
    <span className="font-bold text-sm">{label}</span>
  </button>
);

const MobileNavItem = ({ icon: Icon, label, active, onClick }: any) => (
  <button onClick={onClick} className={`flex flex-col items-center justify-center gap-1 min-w-[64px] transition-all ${active ? 'text-slate-900 dark:text-slate-50' : 'text-slate-400'}`}>
    <div className={`p-2 rounded-xl transition-all ${active ? 'bg-slate-100 dark:bg-slate-800 scale-110' : ''}`}><Icon size={22} strokeWidth={active ? 2.5 : 2} /></div>
    <span className="text-[10px] font-bold uppercase tracking-widest">{label}</span>
  </button>
);

// --- Landing Page ---

const LandingPage = ({ onStart, theme, toggleTheme }: { onStart: () => void, theme: string, toggleTheme: () => void }) => (
  <div className="min-h-screen bg-white dark:bg-slate-950 flex flex-col items-center justify-center p-8 text-center animate-fadeIn relative">
    <div className="absolute top-8 right-8"><ThemeToggle theme={theme} toggleTheme={toggleTheme} /></div>
    <div className="max-w-xl w-full">
      <div className="inline-flex p-5 bg-slate-50 dark:bg-slate-900 rounded-3xl mb-10 shadow-inner"><BrainCircuit size={48} className="text-slate-900 dark:text-slate-50" strokeWidth={2.5} /></div>
      <h1 className="text-5xl md:text-7xl font-black tracking-tighter text-slate-900 dark:text-slate-50 mb-6 uppercase">SMRUTI</h1>
      <p className="text-lg md:text-xl text-slate-600 dark:text-slate-400 mb-14 leading-relaxed font-medium">The Human–AI Decision Memory Continuity System. <br className="hidden md:block"/><span className="text-slate-400 dark:text-slate-500 italic">Capturing logic paths before they fade.</span></p>
      <Button onClick={onStart} className="px-12 py-5 text-xl mx-auto rounded-2xl w-full sm:w-auto">Initialize Core</Button>
    </div>
  </div>
);

// --- Dashboard ---

const Dashboard = ({ decisions, onRecord, onAsk }: any) => {
  const activeDecisions = decisions.filter((d: any) => d.status === 'active');
  const recent = [...activeDecisions].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 4);
  return (
    <div className="space-y-8 max-w-5xl mx-auto w-full pb-32 md:pb-0">
      <header className="px-2"><h2 className="text-3xl md:text-4xl font-black tracking-tight text-slate-900 dark:text-slate-50">Memory Hub</h2><p className="text-slate-500 dark:text-slate-400 mt-2 text-base font-medium">Tracking the logic of your path.</p></header>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 md:gap-6 px-2">
        <StatCard label="Active Logic" value={activeDecisions.length} icon={BrainCircuit} />
        <StatCard label="Archived Paths" value={decisions.length - activeDecisions.length} icon={History} />
        <StatCard label="Core Integrity" value="Safe" icon={CheckCircle2} color="text-emerald-600 dark:text-emerald-400" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8 px-2">
        <Card className="p-6 md:p-8">
          <div className="flex justify-between items-center mb-8"><h3 className="font-black text-slate-900 dark:text-slate-50 uppercase tracking-[0.2em] text-[10px]">Logical Trace</h3><Button variant="ghost" onClick={onRecord} icon={Plus} className="text-[10px] py-1 px-2.5">Add Path</Button></div>
          <div className="space-y-4">{recent.length > 0 ? recent.map((d: any) => (
            <div key={d.id} className="flex items-center justify-between p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 transition-all cursor-pointer group">
              <div className="min-w-0 flex-1"><h4 className="font-bold text-slate-900 dark:text-slate-50 truncate pr-4 text-sm">{d.title}</h4><p className="text-[9px] text-slate-400 uppercase tracking-widest mt-1">v{d.version} • {new Date(d.updatedAt).toLocaleDateString()}</p></div>
              <ChevronRight size={16} className="text-slate-300 group-hover:text-slate-900 dark:group-hover:text-slate-50" />
            </div>
          )) : <p className="text-slate-400 text-center py-12 text-sm italic">Core empty.</p>}</div>
        </Card>
        <Card className="p-8 md:p-10 bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-950 flex flex-col justify-between relative overflow-hidden shadow-2xl">
          <div className="absolute top-0 right-0 p-10 opacity-10 dark:opacity-20"><BrainCircuit size={160} /></div>
          <div className="relative z-10"><h3 className="font-black uppercase tracking-[0.3em] text-[10px] mb-8 opacity-50">Inquiry Engine</h3><p className="text-slate-200 dark:text-slate-800 text-xl md:text-2xl leading-relaxed mb-10 font-serif italic">"Ask to find the reasoning hidden in your past."</p></div>
          <Button onClick={onAsk} variant="secondary" className="w-full py-4 relative z-10 font-bold rounded-xl" icon={BrainCircuit}>Engage Inquiry</Button>
        </Card>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, icon: Icon, color = 'text-slate-900 dark:text-slate-50' }: any) => (
  <Card className="p-6 flex items-center gap-5 border-b-4 border-b-slate-900/5 dark:border-b-white/5"><div className="p-3 bg-slate-50 dark:bg-slate-950 rounded-xl shrink-0"><Icon size={20} className={`${color}`} strokeWidth={2.5} /></div><div><p className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-400 mb-1">{label}</p><p className={`text-2xl font-black tracking-tight ${color}`}>{value}</p></div></Card>
);

const RecordDecision = ({ onSave }: any) => {
  const [formData, setFormData] = useState({ title: '', category: '', intent: '', alternatives: '', finalDecision: '', rejectedReasons: '', constraints: '', confidence: 'Medium', reasoning: '' });
  const handleSubmit = (e: any) => {
    e.preventDefault();
    onSave({ ...formData, constraints: formData.constraints.split(',').map(s => s.trim()).filter(Boolean), alternatives: formData.alternatives.split(',').map(s => s.trim()).filter(Boolean), rejectedReasons: formData.rejectedReasons.split(',').map(s => s.trim()).filter(Boolean) });
  };
  return (
    <div className="max-w-2xl mx-auto pb-32 md:pb-16 px-2 animate-fadeIn">
      <header className="mb-10 text-center"><h2 className="text-3xl md:text-5xl font-black text-slate-900 dark:text-slate-50 tracking-tight uppercase">Trace Path</h2><p className="text-slate-500 dark:text-slate-400 mt-2 font-serif italic">Formalize the memory of your choice.</p></header>
      <form onSubmit={handleSubmit} className="space-y-6">
        <Input label="Title" value={formData.title} onChange={(v: string) => setFormData({...formData, title: v})} required placeholder="The decision subject" />
        <TextArea label="Context & Reasoning" value={formData.reasoning} onChange={(v: string) => setFormData({...formData, reasoning: v})} rows={4} placeholder="Why are you deciding this?" />
        <Input label="Final Path" value={formData.finalDecision} onChange={(v: string) => setFormData({...formData, finalDecision: v})} required placeholder="The choice made" />
        <Button type="submit" className="w-full py-5 text-lg font-black shadow-xl rounded-2xl" icon={Save}>Commit to Memory</Button>
      </form>
    </div>
  );
};

const Input = ({ label, value, onChange, placeholder, required }: any) => (
  <div className="flex flex-col gap-2"><label className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 ml-1">{label}</label><input required={required} className="px-5 py-3.5 bg-slate-50 dark:bg-slate-950/50 border border-slate-200 dark:border-slate-800 rounded-xl focus:border-slate-900 dark:focus:border-slate-100 focus:outline-none text-slate-900 dark:text-slate-50" placeholder={placeholder} value={value} onChange={e => onChange(e.target.value)} /></div>
);

const TextArea = ({ label, value, onChange, placeholder, required, rows = 3 }: any) => (
  <div className="flex flex-col gap-2"><label className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 ml-1">{label}</label><textarea required={required} rows={rows} className="px-5 py-3.5 bg-slate-50 dark:bg-slate-950/50 border border-slate-200 dark:border-slate-800 rounded-xl focus:border-slate-900 dark:focus:border-slate-100 focus:outline-none resize-none text-slate-900 dark:text-slate-50" placeholder={placeholder} value={value} onChange={e => onChange(e.target.value)} /></div>
);

const Timeline = ({ decisions, onViewHistory }: any) => {
  const active = useMemo(() => decisions.filter((d: any) => d.status === 'active').sort((a: any, b: any) => b.updatedAt - a.updatedAt), [decisions]);
  return (
    <div className="space-y-8 max-w-4xl mx-auto w-full px-2 pb-32 md:pb-12">
      <header><h2 className="text-3xl font-black text-slate-900 dark:text-slate-50 tracking-tight uppercase">Logic Timeline</h2></header>
      <div className="space-y-6">{active.length > 0 ? active.map((d: any) => (
          <Card key={d.id} className="p-6 md:p-8 hover:shadow-xl hover:-translate-y-1 transition-all cursor-pointer" onClick={() => onViewHistory(d.rootId)}>
            <div className="flex justify-between items-start mb-6">
              <div><h3 className="text-xl font-black">{d.title}</h3><p className="text-[9px] font-black uppercase text-slate-400">PATH_v{d.version} • {new Date(d.updatedAt).toLocaleDateString()}</p></div>
              <Button variant="secondary" onClick={(e: any) => { e.stopPropagation(); onViewHistory(d.rootId); }} icon={History} className="text-[9px] py-1.5">View Trace</Button>
            </div>
            <div className="p-5 bg-slate-50 dark:bg-slate-950 rounded-xl border-l-4 border-slate-900 dark:border-slate-50 italic font-serif text-sm">"{d.finalDecision}"</div>
          </Card>
        )) : <div className="text-center py-20 italic text-slate-400 text-sm">No memory paths found.</div>}</div>
    </div>
  );
};

const HistoryModal = ({ rootId, decisions, onClose }: any) => {
  const versions = decisions.filter((d: any) => d.rootId === rootId).sort((a: any, b: any) => b.version - a.version);
  return (
    <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-xl z-[100] flex items-center justify-center p-4">
      <div className="w-full max-w-4xl max-h-[85vh] flex flex-col bg-white dark:bg-slate-950 rounded-[2rem] shadow-2xl overflow-hidden">
        <div className="p-6 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center bg-slate-50/50 dark:bg-slate-900/50">
          <h3 className="text-xl font-black text-slate-900 dark:text-slate-50">Memory Evolution</h3>
          <button onClick={onClose} className="p-3 bg-white dark:bg-slate-900 rounded-xl shadow-sm"><X size={20}/></button>
        </div>
        <div className="flex-1 overflow-y-auto p-6 md:p-12 space-y-12">{versions.map((v: any) => (
            <div key={v.id} className="border-l-4 border-slate-100 dark:border-slate-800 pl-6 md:pl-10 relative">
              <div className="absolute left-[-11px] top-0 w-5 h-5 bg-slate-900 dark:bg-white rounded-full flex items-center justify-center text-[8px] font-black text-white dark:text-slate-900">{v.version}</div>
              <h4 className="font-black text-xl mb-4">Version {v.version}</h4>
              <div className="bg-slate-50/80 dark:bg-slate-900/80 p-6 rounded-2xl italic font-serif text-base md:text-lg mb-6 leading-relaxed">"{v.finalDecision}"</div>
              <div className="prose dark:prose-invert max-w-none text-sm md:text-base"><AnimatedMarkdown content={v.reasoning} intent="assist" /></div>
            </div>
          ))}</div>
      </div>
    </div>
  );
};

// --- Main App Root ---

const App = () => {
  const [started, setStarted] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [historyRootId, setHistoryRootId] = useState<string | null>(null);
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  
  useEffect(() => { document.documentElement.classList.toggle('dark', theme === 'dark'); localStorage.setItem('theme', theme); }, [theme]);
  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');
  useEffect(() => { loadDecisions(); }, []);
  
  const loadDecisions = async () => setDecisions(await storage.getAllDecisions());
  
  const handleSave = async (data: any) => { 
    const rootId = Math.random().toString(36).substring(2, 11); 
    const d: Decision = { ...data, id: `${rootId}_v1`, rootId, version: 1, status: 'active', createdAt: Date.now(), updatedAt: Date.now(), updatedBy: 'user' }; 
    await storage.saveDecision(d); await loadDecisions(); setActiveTab('timeline'); 
  };
  
  const handleUpdate = async (change: any) => { 
    const current = decisions.find(d => d.rootId === change.rootId && d.status === 'active'); 
    if (!current) return; 
    await storage.saveDecision({ ...current, status: 'archived', updatedAt: Date.now() }); 
    const next: Decision = { ...current, ...change, id: `${current.rootId}_v${current.version + 1}`, version: current.version + 1, status: 'active', updatedAt: Date.now(), updatedBy: 'ai' }; 
    await storage.saveDecision(next); await loadDecisions(); setActiveTab('timeline'); setHistoryRootId(current.rootId); 
  };
  
  if (!started) return <LandingPage onStart={() => setStarted(true)} theme={theme} toggleTheme={toggleTheme} />;
  
  return (
    <div className="flex flex-col md:flex-row bg-white dark:bg-slate-950 h-screen overflow-hidden font-sans transition-colors duration-500 selection:bg-slate-900 dark:selection:bg-slate-100 selection:text-white dark:selection:text-slate-900">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} theme={theme} toggleTheme={toggleTheme} />
      <main className="flex-1 h-screen overflow-hidden flex flex-col relative bg-white dark:bg-slate-950">
        <div className="flex-1 overflow-y-auto w-full p-4 sm:p-10 md:p-16 no-scrollbar">
          <div className="max-w-7xl mx-auto h-full">
            {activeTab === 'dashboard' && <Dashboard decisions={decisions} onRecord={() => setActiveTab('new')} onAsk={() => setActiveTab('ask')} />}
            {activeTab === 'new' && <RecordDecision onSave={handleSave} />}
            {activeTab === 'timeline' && <Timeline decisions={decisions} onViewHistory={setHistoryRootId} />}
            {activeTab === 'ask' && <AskSmruti decisions={decisions} onUpdate={handleUpdate} />}
          </div>
        </div>
      </main>
      <MobileNavbar activeTab={activeTab} setActiveTab={setActiveTab} />
      {historyRootId && <HistoryModal rootId={historyRootId} decisions={decisions} onClose={() => setHistoryRootId(null)} />}
    </div>
  );
};

const rootEl = document.getElementById('root');
if (rootEl) createRoot(rootEl).render(<App />);
