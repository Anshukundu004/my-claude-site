import React, { useState, useEffect, useCallback, useMemo } from 'react';

// ====================================================================
// 1. Interfaces & Types (Crucial for TypeScript)
// ====================================================================

// General props for animated demos
interface DemoProps {
  isPlaying: boolean;
  speed: number;
}

// Type for a single section in DetailedInfo
interface Section {
  title: string;
  content: string;
}

// Type for DetailedInfo component props
interface DetailedInfoProps {
  title: string;
  sections: Section[];
}

// Types for AIChatAssistant
interface ChatMessage {
  id: number;
  role: 'user' | 'ai';
  text: string;
}

// Types for AgenticAIDemo
interface Thought {
  id: number;
  text: string;
  phase: string;
}

// Types for MultiAgentDemo
interface Agent {
  id: number;
  x: number;
  y: number;
  role: string;
  color: string;
  target: null | string;
}

interface Message {
  id: number;
  from: string;
  to: string;
  text: string;
}

// ====================================================================
// 2. Helper & Utility Components
// ====================================================================

// DetailedInfo Component (Corrected with Types and Enhanced Rendering)
const DetailedInfo: React.FC<DetailedInfoProps> = ({ title, sections }) => {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(() => new Set([0]));

  const toggleSection = useCallback((index: number) => {
    setExpandedSections(prevExpanded => {
      const newExpanded = new Set(prevExpanded);
      if (newExpanded.has(index)) {
        newExpanded.delete(index);
      } else {
        newExpanded.add(index);
      }
      return newExpanded;
    });
  }, []);

  const expandAll = useCallback(() => {
    setExpandedSections(new Set(sections.map((_, i) => i)));
  }, [sections]);

  const collapseAll = useCallback(() => {
    setExpandedSections(new Set());
  }, []);

  // Function to handle custom Markdown-like formatting (bolding and code blocks)
  const renderContent = (content: string) => {
    return content.split('\n').map((paragraph, pIndex) => {
      // 1. Handle code blocks (lines starting with # or import)
      if (paragraph.trim().startsWith('#') || paragraph.trim().startsWith('import')) {
        return (
          <pre key={pIndex} className="bg-black/30 p-4 rounded-lg overflow-x-auto mb-3">
            <code className="text-green-300 text-sm font-mono">
              {paragraph}
            </code>
          </pre>
        );
      }
      
      // 2. Handle bold text with **
      if (paragraph.includes('**')) {
        const parts = paragraph.split('**');
        const elements = parts.map((part, partIndex) =>
          partIndex % 2 === 1 ? (
            <strong key={partIndex} className="text-purple-300 font-semibold">
              {part}
            </strong>
          ) : (
            <span key={partIndex}>{part}</span>
          )
        );
        return (
          <p key={pIndex} className="mb-3">
            {elements}
          </p>
        );
      }
      
      // 3. Regular paragraphs
      if (paragraph.trim()) {
        return (
          <p key={pIndex} className="mb-3">
            {paragraph}
          </p>
        );
      }
      return null;
    });
  };

  return (
    <div className="mt-8 bg-gradient-to-br from-white/5 to-white/10 rounded-xl p-6 border border-white/20">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
          {title}
        </h3>
        <div className="flex gap-2">
          <button
            onClick={expandAll}
            className="text-xs bg-white/10 hover:bg-white/20 px-3 py-1 rounded-lg transition-all"
          >
            Expand All
          </button>
          <button
            onClick={collapseAll}
            className="text-xs bg-white/10 hover:bg-white/20 px-3 py-1 rounded-lg transition-all"
          >
            Collapse All
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {sections.map((section, index) => {
          const isExpanded = expandedSections.has(index);
          return (
            <div
              key={index}
              className="bg-white/5 rounded-lg border border-white/10 overflow-hidden transition-all"
            >
              <button
                onClick={() => toggleSection(index)}
                className="w-full px-5 py-4 flex justify-between items-center hover:bg-white/10 transition-all text-left"
              >
                <span className="font-semibold text-lg text-cyan-300">
                  {section.title}
                </span>
                <span className="text-2xl text-cyan-400">
                  {isExpanded ? '−' : '+'}
                </span>
              </button>

              {isExpanded && (
                <div className="px-5 pb-5 pt-2">
                  <div className="text-gray-300 leading-relaxed">
                    {renderContent(section.content)}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};


// ====================================================================
// 3. Demo Components (The Core Content)
// ====================================================================

// 3.1 AIConceptsDemo
const AIConceptsDemo: React.FC = () => {
  const concepts: { title: string; desc: string; icon: string; }[] = useMemo(() => ([
    { title: "Machine Learning", desc: "Systems learn from data without explicit programming.", icon: "🧠" },
    { title: "Deep Learning", desc: "ML using neural networks with multiple layers.", icon: "🧱" },
    { title: "Reinforcement Learning", desc: "Agent learns via trial-and-error using rewards.", icon: "🏆" },
    { title: "Generative AI", desc: "Creates new content (text, images, code).", icon: "✨" },
    { title: "Agentic AI", desc: "Autonomous system that plans and acts to achieve goals.", icon: "🤖" },
    { title: "LLMs", desc: "Large Language Models, foundation for many AI apps.", icon: "📜" },
  ]), []);

  return (
    <div className="p-8">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
        {concepts.map((concept, index) => (
          <div key={index} className="bg-white/10 p-6 rounded-xl shadow-lg hover:bg-white/20 transition duration-300">
            <div className="text-3xl mb-3">{concept.icon}</div>
            <h3 className="text-xl font-bold mb-2 text-cyan-400">{concept.title}</h3>
            <p className="text-sm text-gray-300">{concept.desc}</p>
          </div>
        ))}
      </div>
       <DetailedInfo 
        title="Core AI Concepts" 
        sections={[
          { title: "Machine Learning", content: "**ML** is a subset of AI that enables systems to learn from data. **Supervised Learning** uses labeled data; **Unsupervised Learning** finds patterns in unlabeled data; **Reinforcement Learning** learns via trial and error." },
          { title: "Deep Learning", content: "Uses **Neural Networks** with many hidden layers. Excellent for complex pattern recognition like image, speech, and natural language processing." },
          { title: "LLMs (Large Language Models)", content: "Trained on massive text data to predict the next token. This capability underpins complex reasoning, translation, and code generation. They use the **Transformer Architecture**." },
        ]} 
      />
    </div>
  );
};


// 3.2 AIChatAssistant
const AIChatAssistant: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 1, role: 'ai', text: "Hello! I'm an AI assistant. How can I help you today?" }
  ]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleSend = useCallback(() => {
    if (input.trim() === '') return;

    const userMessage: ChatMessage = { id: Date.now(), role: 'user', text: input.trim() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Simulated AI response
    setTimeout(() => {
      const aiResponse: ChatMessage = { 
        id: Date.now() + 1, 
        role: 'ai', 
        text: `Thank you for your question about "${input.substring(0, 20)}...". As a **Generative AI**, I can tell you that the core of this topic involves complex algorithms and massive data sets. Let me know if you want to dive into specifics!`
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsLoading(false);
    }, 1500);
  }, [input]);

  const chatInfoSections: Section[] = useMemo(() => ([
    { title: "Architecture", content: "Typically uses a **Transformer model** (like GPT or Claude). Consists of an encoder and decoder, or just a decoder stack, processing tokens." },
    { title: "Functionality", content: "Predicts the next most probable token in a sequence based on all prior tokens, generating human-like text responses." },
    { title: "Training", content: "Trained on massive, diverse datasets using unsupervised learning to learn grammar, facts, and reasoning patterns." },
    { title: "Fine-Tuning", content: "Often uses **Reinforcement Learning from Human Feedback (RLHF)** to align output with human values and instructions." },
  ]), []);

  return (
    <div className="p-8">
      <div className="bg-white/5 rounded-xl p-4 flex flex-col h-[500px]">
        <div className="flex-1 overflow-y-auto space-y-4 mb-4">
          {messages.map((msg) => (
            <div 
              key={msg.id} 
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div 
                className={`max-w-xs md:max-w-md p-3 rounded-xl shadow-md ${
                  msg.role === 'user' ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-100'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-700 text-gray-100 p-3 rounded-xl shadow-md animate-pulse">
                AI is thinking...
              </div>
            </div>
          )}
        </div>
        <div className="flex">
          <input
            type="text"
            className="flex-1 p-3 rounded-l-xl bg-gray-800 text-white border-2 border-r-0 border-cyan-600 focus:outline-none"
            placeholder="Ask the AI a question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            className="p-3 rounded-r-xl bg-cyan-600 text-white font-bold hover:bg-cyan-700 transition duration-200 disabled:opacity-50"
            disabled={isLoading}
          >
            Send
          </button>
        </div>
      </div>
      <DetailedInfo title="LLM Architecture & Training" sections={chatInfoSections} />
    </div>
  );
};


// 3.3 NeuralNetworkDemo
const NeuralNetworkDemo: React.FC<DemoProps> = ({ isPlaying, speed }) => {
  const [weights, setWeights] = useState<number[]>([0.5, -0.3, 0.9]);
  const [output, setOutput] = useState<number>(0);
  const [inputs, setInputs] = useState<number[]>([1.0, 0.5, -0.7]);

  const processNeuron = useCallback(() => {
    const weightedSum: number = inputs.reduce((sum: number, input: number, index: number) => 
      sum + input * weights[index], 0); 
    
    // Sigmoid activation function
    const newOutput: number = 1 / (1 + Math.exp(-weightedSum));
    setOutput(newOutput);
    
    // Simulate learning (adjusting weights)
    setWeights(prev => prev.map(w => w + (Math.random() - 0.5) * 0.05));
    // Simulate new input data
    setInputs(prev => prev.map(() => Math.random() * 2 - 1));
  }, [weights, inputs]);

  useEffect(() => {
    if (!isPlaying) return;

    const interval: number = setInterval(() => {
      processNeuron();
    }, 1000 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, speed, processNeuron]);

  const nnInfoSections: Section[] = useMemo(() => ([
    { title: "Neuron", content: "The basic unit. It receives inputs, computes a **weighted sum**, adds a bias, and passes the result through an **activation function**." },
    { title: "Layers", content: "Input, Hidden, and Output layers. **Deep Learning** means many hidden layers." },
    { title: "Weights & Bias", content: "Parameters learned during training. Weights determine input importance; bias shifts the activation function." },
    { title: "Backpropagation", content: "The core training algorithm. It calculates the **gradient** of the loss function with respect to the weights and adjusts them to minimize error." },
  ]), []);
  
  return (
    <div className="p-8">
        <div className="flex justify-center items-center h-48 space-x-12">
            {/* Inputs */}
            <div className="flex flex-col space-y-4">
                {inputs.map((input, index) => (
                    <div key={index} className="flex items-center space-x-2">
                        <div className="text-lg font-mono">Input {index + 1}:</div>
                        <div className="text-2xl font-bold text-green-400">{input.toFixed(2)}</div>
                    </div>
                ))}
            </div>

            {/* Neuron */}
            <div className="relative w-24 h-24 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-full flex items-center justify-center shadow-xl">
                <div className="text-4xl font-bold">Σ</div>
            </div>

            {/* Output */}
            <div className="flex flex-col items-start space-y-4">
                <div className="text-lg font-mono">Output (Sigmoid):</div>
                <div className="text-4xl font-extrabold text-pink-400">{output.toFixed(4)}</div>
            </div>
        </div>

        <div className="mt-8 text-center">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">Current Weights:</h3>
            <div className="flex justify-center space-x-6">
                {weights.map((w, index) => (
                    <div key={index} className="bg-white/10 p-2 rounded-lg font-mono text-sm">
                        W{index + 1}: <span className={w > 0 ? 'text-green-300' : 'text-red-400'}>{w.toFixed(4)}</span>
                    </div>
                ))}
            </div>
        </div>
        <DetailedInfo title="Deep Learning Fundamentals" sections={nnInfoSections} />
    </div>
  );
};


// 3.4 ReinforcementDemo
const ReinforcementDemo: React.FC<DemoProps> = ({ isPlaying, speed }) => {
    const [reward, setReward] = useState<number>(0);
    const [cumulativeReward, setCumulativeReward] = useState<number>(0);
    const [episode, setEpisode] = useState<number>(1);
    const [action, setAction] = useState<string>('Idle');

    const actions = useMemo(() => ['Explore', 'Exploit', 'Observe', 'Punish', 'Reward'], []);

    const stepSimulation = useCallback(() => {
        const randomAction = actions[Math.floor(Math.random() * actions.length)];
        setAction(randomAction);

        let currentReward: number;
        if (randomAction === 'Reward') {
            currentReward = Math.random() * 5 + 1; // +1 to +6
        } else if (randomAction === 'Punish') {
            currentReward = -(Math.random() * 5 + 1); // -1 to -6
        } else if (randomAction === 'Exploit') {
            currentReward = 0.5; // Small positive reward for known good action
        } else {
            currentReward = Math.random() * 1 - 0.5; // Small, random
        }

        setReward(currentReward);
        setCumulativeReward(prev => prev + currentReward);
        setEpisode(prev => prev + 1);

    }, [actions]);

    useEffect(() => {
        if (!isPlaying) return;

        const interval: number = setInterval(() => {
            stepSimulation();
        }, 800 / speed);

        return () => clearInterval(interval);
    }, [isPlaying, speed, stepSimulation]);

    const rlInfoSections: Section[] = useMemo(() => ([
        { title: "Markov Decision Process (MDP)", content: "Formal framework: **State, Action, Transition Probability, Reward**. The environment satisfies the Markov Property." },
        { title: "Q-Learning", content: "A value-based, off-policy algorithm that learns an action-value function **Q(s, a)**, estimating the expected return for taking action 'a' in state 's'." },
        { title: "Policy Gradient", content: "Directly optimizes the policy function **π(a|s)**. Examples include REINFORCE and PPO (Proximal Policy Optimization)." },
        { title: "Exploration vs Exploitation", content: "The Dilemma: Should agent try new actions (**explore**) or use best known actions (**exploit**)? Strategies include ε-Greedy." },
        { title: "Real-World Applications", content: "Includes robotics (locomotion, manipulation), game playing (**AlphaGo**), autonomous vehicles, and recommendation systems." },
    ]), []);

    return (
        <div className="p-8">
            <div className="flex justify-around items-center text-center">
                <div className="bg-white/10 p-6 rounded-xl w-48">
                    <div className="text-sm text-gray-400">Episode</div>
                    <div className="text-3xl font-bold text-cyan-400">{episode}</div>
                </div>
                <div className="text-4xl">→</div>
                <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-6 rounded-full w-48 h-48 flex flex-col items-center justify-center shadow-2xl">
                    <div className="text-sm text-white/70">Agent Action</div>
                    <div className="text-2xl font-extrabold mt-1">{action}</div>
                </div>
                <div className="text-4xl">→</div>
                <div className="bg-white/10 p-6 rounded-xl w-48">
                    <div className="text-sm text-gray-400">Instant Reward</div>
                    <div className={`text-3xl font-bold ${reward > 0 ? 'text-green-400' : reward < 0 ? 'text-red-400' : 'text-yellow-400'}`}>{reward.toFixed(2)}</div>
                </div>
            </div>
            <div className="mt-8 text-center">
                <h3 className="text-lg font-semibold text-purple-400 mb-2">Cumulative Reward</h3>
                <div className={`text-5xl font-extrabold ${cumulativeReward > 0 ? 'text-green-500' : cumulativeReward < 0 ? 'text-red-500' : 'text-gray-400'}`}>
                    {cumulativeReward.toFixed(2)}
                </div>
            </div>
            <DetailedInfo 
                title="Reinforcement Learning Fundamentals"
                sections={rlInfoSections}
            />
        </div>
    );
};


// 3.5 AgenticAIDemo
const AgenticAIDemo: React.FC<DemoProps> = ({ isPlaying, speed }) => {
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [thoughts, setThoughts] = useState<Thought[]>([]);

  const steps = useMemo(() => ([
    { phase: 'Perceive', desc: 'Analyze environment & goals', color: 'from-cyan-500 to-blue-500', icon: '👁️' },
    { phase: 'Reason', desc: 'Plan sequence of actions', color: 'from-purple-500 to-pink-500', icon: '🧠' },
    { phase: 'Act', desc: 'Execute chosen action', color: 'from-green-500 to-emerald-500', icon: '⚡' },
    { phase: 'Learn', desc: 'Update knowledge from feedback', color: 'from-orange-500 to-red-500', icon: '📚' }
  ]), []);

  useEffect(() => {
    if (!isPlaying) return;

    const interval: number = setInterval(() => {
      setCurrentStep(prev => (prev + 1) % steps.length);

      const nextStepIndex = (currentStep + 1) % steps.length;
      const newThought: Thought = {
        id: Date.now(),
        text: steps[nextStepIndex].desc,
        phase: steps[nextStepIndex].phase
      };

      setThoughts(prev => [...prev.slice(-3), newThought]);
    }, 2000 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, speed, currentStep, steps]); 

  const agentInfoSections: Section[] = useMemo(() => ([
    { title: "What is Agentic AI?", content: "Autonomous systems capable of perceiving their environment, making **multi-step decisions**, taking actions (often with tools), and learning from outcomes to achieve specific goals without constant human supervision." },
    { title: "Key Characteristics", content: "**Autonomy**: Operates independently. **Goal-Oriented**: Driven by objectives. **Tool Use**: Leverages APIs, databases, or code to extend capabilities." },
    { title: "Core Components", content: "Perception Module (input), **Reasoning Engine (LLM planning)**, Action Execution (tools), and Memory Systems (context)." },
    { title: "Agentic AI Frameworks", content: "Patterns like **ReAct (Reason + Act)** which alternates between generating a thought, taking an action, and observing the result. Also AutoGPT and CrewAI." },
  ]), []);

  return (
    <div className="p-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Agentic AI Decision Loop</h2>
        <p className="text-gray-300">Autonomous agents continuously perceive, reason, act, and learn - operating independently to achieve goals.</p>
      </div>

      <div className="flex gap-8 items-center">
        <div className="flex-1">
          <div className="relative w-96 h-96 mx-auto">
            {steps.map((step, idx) => {
              const angle = (idx * 2 * Math.PI) / steps.length - Math.PI / 2;
              const x = Math.cos(angle) * 140 + 192;
              const y = Math.sin(angle) * 140 + 192;
              
              return (
                <div
                  key={idx}
                  className={`absolute w-24 h-24 rounded-full bg-gradient-to-br ${step.color} flex flex-col items-center justify-center transition-all transform ${
                    currentStep === idx ? 'scale-125 shadow-2xl z-10' : 'scale-100 opacity-60'
                  }`}
                  style={{ left: x - 48, top: y - 48 }}
                >
                  <div className="text-3xl mb-1">{step.icon}</div>
                  <div className="text-xs font-bold">{step.phase}</div>
                </div>
              );
            })}
            
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-32 h-32 rounded-full bg-gradient-to-br from-yellow-500 to-orange-500 flex items-center justify-center text-4xl shadow-2xl">
              🤖
            </div>

            {isPlaying && (
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 rounded-full border-4 border-cyan-400 opacity-30 animate-ping" />
            )}
          </div>
        </div>

        <div className="w-80 space-y-3">
          <div className="bg-white/5 p-4 rounded-xl">
            <div className="text-sm text-cyan-400 font-semibold mb-2">Agent Thoughts:</div>
            <div className="space-y-2">
              {thoughts.map((thought, idx) => (
                <div
                  key={thought.id}
                  className="bg-white/10 p-2 rounded text-sm"
                  style={{ opacity: 1 - idx * 0.3 }}
                >
                  <span className="font-semibold text-purple-400">{thought.phase}:</span> {thought.text}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <DetailedInfo 
        title="Agentic AI - Complete Guide"
        sections={agentInfoSections}
      />
    </div>
  );
};


// 3.6 MultiAgentDemo
const MultiAgentDemo: React.FC<DemoProps> = ({ isPlaying, speed }) => {
  const [agents, setAgents] = useState<Agent[]>([
    { id: 1, x: 20, y: 30, role: 'Explorer', color: 'from-blue-500 to-cyan-500', target: null },
    { id: 2, x: 80, y: 30, role: 'Analyzer', color: 'from-purple-500 to-pink-500', target: null },
    { id: 3, x: 50, y: 70, role: 'Executor', color: 'from-green-500 to-emerald-500', target: null }
  ]);
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    if (!isPlaying) return;

    const interval: number = setInterval(() => {
      setAgents(prev => prev.map(agent => ({
        ...agent,
        x: agent.x + (Math.random() - 0.5) * 10,
        y: agent.y + (Math.random() - 0.5) * 10
      })));

      const randomAgentIndex = () => Math.floor(Math.random() * agents.length);
      const newMsg: Message = {
        id: Date.now(),
        from: agents[randomAgentIndex()].role,
        to: agents[randomAgentIndex()].role,
        text: ['Share data', 'Task complete', 'Need assistance', 'Coordinating'][Math.floor(Math.random() * 4)],
      };
      setMessages(prev => [...prev.slice(-4), newMsg]);
      
    }, 1500 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, speed, agents]); 
  
  const multiAgentInfoSections: Section[] = useMemo(() => ([
    { title: "What are Multi-Agent Systems?", content: "Systems where multiple autonomous agents interact and coordinate to achieve individual or collective goals, solving problems too complex for a single agent." },
    { title: "Core Concepts", content: "**Cooperative/Competitive**: Agents can work together or against each other. **Coordination**: Mechanisms like negotiation or planning synchronize actions." },
    { title: "Coordination Mechanisms", content: "Negotiation (**Contract Net Protocol**, Auctions), Centralized/Distributed Planning, Voting, and **BlackBoard** systems for shared knowledge." },
    { title: "Communication Protocols", content: "**FIPA ACL** (Agent Communication Language) is a standard message structure defining performatives (INFORM, REQUEST, PROPOSE) and protocols." },
    { title: "Multi-Agent Learning", content: "Challenges include **non-stationarity** (other agents changing) and **credit assignment** (who gets the reward). Solutions include Opponent Modeling and QMIX." },
  ]), []);

  return (
    <div className="p-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Multi-Agent Collaboration</h2>
        <p className="text-gray-300">Multiple AI agents work together, communicating and coordinating to solve complex tasks beyond single-agent capabilities.</p>
      </div>

      <div className="flex gap-8">
        <div className="flex-1 bg-white/5 rounded-xl p-6 relative h-96">
          {agents.map((agent: Agent) => (
            <div
              key={agent.id}
              className={`absolute w-20 h-20 rounded-full bg-gradient-to-br ${agent.color} flex flex-col items-center justify-center transition-all shadow-lg`}
              style={{
                left: `${Math.max(10, Math.min(85, agent.x))}%`,
                top: `${Math.max(10, Math.min(80, agent.y))}%`
              }}
            >
              <div className="text-2xl">🤖</div>
              <div className="text-xs font-bold mt-1">{agent.role}</div>
            </div>
          ))}

          {isPlaying && messages.length > 0 && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              {agents.map((from, i) => 
                agents.slice(i + 1).map((to, j) => (
                  <line
                    key={`${i}-${j}`}
                    x1={`${from.x}%`}
                    y1={`${from.y}%`}
                    x2={`${to.x}%`}
                    y2={`${to.y}%`}
                    stroke="rgba(139, 92, 246, 0.3)"
                    strokeWidth="2"
                    className="animate-pulse"
                  />
                ))
              )}
            </svg>
          )}
        </div>

        <div className="w-80 space-y-3">
          <div className="bg-white/5 p-4 rounded-xl">
          <div className="text-sm font-semibold text-orange-400 mb-3">Agent Messages:</div>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {messages.map((msg: Message) => (
                <div key={msg.id} className="bg-white/10 p-2 rounded text-xs">
                  <div className="font-semibold text-cyan-400">{msg.from} → {msg.to}</div>
                  <div className="text-gray-300">{msg.text}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <DetailedInfo 
        title="Multi-Agent Systems - Complete Guide"
        sections={multiAgentInfoSections}
      />
    </div>
  );
};
// ====================================================================
// 4. Main App Component (Bringing everything together)
// ====================================================================

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('Concepts');
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);

  const renderDemo = () => {
    const demoProps: DemoProps = { isPlaying, speed };
    switch (activeTab) {
      case 'Concepts':
        return <AIConceptsDemo />;
      case 'Chat':
      return <AIChatAssistant />;
      case 'NeuralNet':
        return <NeuralNetworkDemo {...demoProps} />;
      case 'RL':
        return <ReinforcementDemo {...demoProps} />;
      case 'Agentic':
        return <AgenticAIDemo {...demoProps} />;
      case 'MultiAgent':
        return <MultiAgentDemo {...demoProps} />;
      default:
        return <AIConceptsDemo />;
    }
  };  
  const tabs: { id: string; name: string; icon: string; }[] = [
    { id: 'Concepts', name: 'AI Concepts', icon: '🧠' },
    { id: 'Chat', name: 'Chat Assistant', icon: '💬' },
    { id: 'NeuralNet', name: 'Neural Network', icon: '🧬' },
    { id: 'RL', name: 'Reinforcement L.', icon: '🏆' },
    { id: 'Agentic', name: 'Agentic AI', icon: '🤖' },
    { id: 'MultiAgent', name: 'Multi-Agent Sys', icon: '🤝' },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <header className="text-center py-6 border-b border-gray-700">
        <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600">
          Interactive AI Demo Hub
        </h1>
        <p className="text-gray-400 mt-2">Explore the core concepts of Modern Artificial Intelligence</p>
      </header>

      {/* Tabs */}
      <div className="flex justify-center flex-wrap gap-2 py-4 border-b border-gray-700">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg transition duration-200 font-semibold flex items-center space-x-2 ${
              activeTab === tab.id
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.name}</span>
          </button>
        ))}
      </div>

      {/* Controls */}
      {['NeuralNet', 'RL', 'Agentic', 'MultiAgent'].includes(activeTab) && (
        <div className="flex justify-center items-center py-4 space-x-6 bg-gray-800/50 rounded-lg my-4 mx-auto max-w-lg">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-2 rounded-full bg-cyan-600 text-white hover:bg-cyan-700 transition"
            title={isPlaying ? 'Pause Simulation' : 'Start Simulation'}
          >
            {isPlaying ? '⏸️' : '▶️'}
          </button>
          <label className="text-gray-300 flex items-center space-x-2">
            <span>Speed: {speed}x</span>
            <input
              type="range"
              min="0.5"
              max="3"
              step="0.5"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-32"
            />
          </label>
        </div>
      )}

      {/* Content */}
      <main className="bg-gray-800 rounded-xl shadow-2xl mt-4">
        {renderDemo()}
      </main>

      <footer className="text-center text-gray-500 text-sm mt-8 py-4 border-t border-gray-700">
        © 2024 AI Demo Hub. Created with React & TypeScript.
      </footer>
    </div>
  );
};

export default App; // Changed export to App




