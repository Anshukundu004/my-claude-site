import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Zap, Brain, Users, Target, MessageSquare, Save, Sparkles } from 'lucide-react';

const AIConceptsDemo = () => {
  const [activeDemo, setActiveDemo] = useState('neural-network');
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [userProgress, setUserProgress] = useState({});
  const [aiChatOpen, setAiChatOpen] = useState(false);

  // Load user progress on mount
  useEffect(() => {
    loadProgress();
  }, []);

  const loadProgress = async () => {
    try {
      const result = await window.storage.get('user-progress');
      if (result) {
        setUserProgress(JSON.parse(result.value));
      }
    } catch (error) {
      console.log('No saved progress yet');
    }
  };

  const saveProgress = async (demoId) => {
    try {
      const newProgress = {
        ...userProgress,
        [demoId]: {
          completed: true,
          timestamp: new Date().toISOString(),
          viewCount: (userProgress[demoId]?.viewCount || 0) + 1
        }
      };
      await window.storage.set('user-progress', JSON.stringify(newProgress));
      setUserProgress(newProgress);
    } catch (error) {
      console.error('Failed to save progress:', error);
    }
  };

  const demos = [
    { id: 'neural-network', name: 'Neural Network', icon: Brain, color: 'from-blue-500 to-cyan-500' },
    { id: 'reinforcement', name: 'Reinforcement Learning', icon: Target, color: 'from-purple-500 to-pink-500' },
    { id: 'agentic', name: 'Agentic AI', icon: Zap, color: 'from-green-500 to-emerald-500' },
    { id: 'multi-agent', name: 'Multi-Agent Systems', icon: Users, color: 'from-orange-500 to-red-500' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <div className="bg-black/30 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-6 flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              Interactive AI Concepts Explorer
            </h1>
            <p className="text-gray-300 mt-2">Visualize and understand core AI concepts with real AI assistance</p>
          </div>
          <button
            onClick={() => setAiChatOpen(!aiChatOpen)}
            className="bg-gradient-to-r from-cyan-500 to-purple-500 px-6 py-3 rounded-xl flex items-center gap-2 hover:shadow-xl transition-all"
          >
            <Sparkles className="w-5 h-5" />
            Ask AI
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="bg-white/5 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-300">Your Learning Progress</span>
            <span className="text-sm font-bold text-cyan-400">
              {Object.keys(userProgress).length} / {demos.length} Demos Explored
            </span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-cyan-500 to-purple-500 h-2 rounded-full transition-all"
              style={{ width: `${(Object.keys(userProgress).length / demos.length) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {demos.map((demo) => {
            const Icon = demo.icon;
            const isCompleted = userProgress[demo.id]?.completed;
            return (
              <button
                key={demo.id}
                onClick={() => {
                  setActiveDemo(demo.id);
                  setIsPlaying(false);
                  saveProgress(demo.id);
                }}
                className={`p-4 rounded-xl transition-all relative ${
                  activeDemo === demo.id
                    ? `bg-gradient-to-r ${demo.color} shadow-lg scale-105`
                    : 'bg-white/5 hover:bg-white/10'
                }`}
              >
                {isCompleted && (
                  <div className="absolute -top-2 -right-2 bg-green-500 rounded-full w-6 h-6 flex items-center justify-center text-xs">
                    ✓
                  </div>
                )}
                <Icon className="w-6 h-6 mx-auto mb-2" />
                <div className="text-sm font-medium">{demo.name}</div>
                {userProgress[demo.id]?.viewCount > 0 && (
                  <div className="text-xs text-gray-400 mt-1">
                    Viewed {userProgress[demo.id].viewCount}x
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Demo Area */}
      <div className="max-w-7xl mx-auto px-6 pb-12">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl border border-white/10 overflow-hidden">
          {/* Controls */}
          <div className="bg-black/30 p-4 flex items-center justify-between border-b border-white/10">
            <div className="flex gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 rounded-lg flex items-center gap-2 hover:shadow-lg transition-all"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <button
                onClick={() => setIsPlaying(false)}
                className="bg-white/10 px-4 py-2 rounded-lg hover:bg-white/20 transition-all"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-300">Speed:</span>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.5"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="w-32"
              />
              <span className="text-sm font-mono">{speed}x</span>
            </div>
          </div>

          {/* Demo Content */}
          {activeDemo === 'neural-network' && (
            <NeuralNetworkDemo isPlaying={isPlaying} speed={speed} />
          )}
          {activeDemo === 'reinforcement' && (
            <ReinforcementDemo isPlaying={isPlaying} speed={speed} />
          )}
          {activeDemo === 'agentic' && (
            <AgenticAIDemo isPlaying={isPlaying} speed={speed} />
          )}
          {activeDemo === 'multi-agent' && (
            <MultiAgentDemo isPlaying={isPlaying} speed={speed} />
          )}
        </div>
      </div>

      {/* AI Chat Assistant */}
      {aiChatOpen && (
        <AIChatAssistant 
          onClose={() => setAiChatOpen(false)}
          currentDemo={demos.find(d => d.id === activeDemo)?.name}
        />
      )}
    </div>
  );
};

// AI Chat Assistant Component
const AIChatAssistant = ({ onClose, currentDemo }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Load chat history
    loadChatHistory();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadChatHistory = async () => {
    try {
      const result = await window.storage.get('chat-history');
      if (result) {
        setMessages(JSON.parse(result.value));
      }
    } catch (error) {
      console.log('No chat history yet');
    }
  };

  const saveChatHistory = async (newMessages) => {
    try {
      await window.storage.set('chat-history', JSON.stringify(newMessages));
    } catch (error) {
      console.error('Failed to save chat:', error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1000,
          system: `You are an AI expert helping users understand AI concepts. The user is currently viewing the "${currentDemo}" demo. Provide clear, helpful explanations suitable for both beginners and technical users. Keep responses concise and engaging.`,
          messages: newMessages
        })
      });

      const data = await response.json();
      const assistantMessage = {
        role: 'assistant',
        content: data.content[0].text
      };

      const updatedMessages = [...newMessages, assistantMessage];
      setMessages(updatedMessages);
      saveChatHistory(updatedMessages);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again!'
      };
      setMessages([...newMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 rounded-2xl shadow-2xl w-full max-w-2xl h-[600px] flex flex-col border border-white/10">
        {/* Header */}
        <div className="bg-gradient-to-r from-cyan-500 to-purple-500 p-4 rounded-t-2xl flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5" />
            <h3 className="font-bold">AI Learning Assistant</h3>
          </div>
          <button
            onClick={onClose}
            className="hover:bg-white/20 rounded-lg p-1 transition-all"
          >
            ✕
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-8">
              <Sparkles className="w-12 h-12 mx-auto mb-4 text-purple-400" />
              <p>Ask me anything about {currentDemo} or AI concepts!</p>
              <div className="mt-4 space-y-2">
                <button
                  onClick={() => setInput('Explain this concept in simple terms')}
                  className="block w-full bg-white/5 hover:bg-white/10 p-2 rounded-lg text-sm transition-all"
                >
                  💡 Explain this concept in simple terms
                </button>
                <button
                  onClick={() => setInput('What are real-world applications?')}
                  className="block w-full bg-white/5 hover:bg-white/10 p-2 rounded-lg text-sm transition-all"
                >
                  🌍 What are real-world applications?
                </button>
                <button
                  onClick={() => setInput('How can I learn more about this?')}
                  className="block w-full bg-white/5 hover:bg-white/10 p-2 rounded-lg text-sm transition-all"
                >
                  📚 How can I learn more about this?
                </button>
              </div>
            </div>
          )}
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-3 ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500'
                    : 'bg-white/10'
                }`}
              >
                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white/10 rounded-lg p-3">
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-white/10">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Ask about AI concepts..."
              className="flex-1 bg-white/10 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !input.trim()}
              className="bg-gradient-to-r from-cyan-500 to-purple-500 px-6 py-2 rounded-lg hover:shadow-lg transition-all disabled:opacity-50"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Neural Network Demo (keeping original)
const NeuralNetworkDemo = ({ isPlaying, speed }) => {
  const [activations, setActivations] = useState([]);

  useEffect(() => {
    if (!isPlaying) return;
    
    const interval = setInterval(() => {
      setActivations(prev => {
        const newAct = { id: Date.now(), progress: 0 };
        return [...prev.slice(-5), newAct];
      });
    }, 1500 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, speed]);

  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setActivations(prev => 
        prev.map(a => ({ ...a, progress: Math.min(a.progress + 0.02 * speed, 1) }))
          .filter(a => a.progress < 1)
      );
    }, 30);

    return () => clearInterval(interval);
  }, [isPlaying, speed]);

  const layers = [
    { name: 'Input Layer', neurons: 4, color: 'bg-cyan-500' },
    { name: 'Hidden Layer 1', neurons: 6, color: 'bg-purple-500' },
    { name: 'Hidden Layer 2', neurons: 5, color: 'bg-pink-500' },
    { name: 'Output Layer', neurons: 3, color: 'bg-green-500' }
  ];

  return (
    <div className="p-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Neural Network Forward Propagation</h2>
        <p className="text-gray-300">Watch how data flows through layers of interconnected neurons, with each connection having a weight that influences the final output.</p>
      </div>

      <div className="flex justify-around items-center min-h-[400px] relative">
        {layers.map((layer, layerIdx) => (
          <div key={layerIdx} className="flex flex-col items-center gap-6">
            <div className="text-sm font-semibold text-gray-300 mb-2">{layer.name}</div>
            <div className="flex flex-col gap-4">
              {Array.from({ length: layer.neurons }).map((_, neuronIdx) => (
                <div
                  key={neuronIdx}
                  className={`w-12 h-12 rounded-full ${layer.color} flex items-center justify-center font-bold shadow-lg transition-all ${
                    isPlaying && Math.random() > 0.5 ? 'scale-110 shadow-2xl' : ''
                  }`}
                  style={{
                    opacity: isPlaying ? 0.5 + Math.random() * 0.5 : 0.7
                  }}
                >
                  {neuronIdx + 1}
                </div>
              ))}
            </div>
          </div>
        ))}

        {activations.map(act => (
          <div
            key={act.id}
            className="absolute top-1/2 left-0 w-full h-1 bg-gradient-to-r from-cyan-400 to-green-400 transform -translate-y-1/2 opacity-50"
            style={{
              width: `${act.progress * 100}%`,
              transition: 'width 0.03s linear'
            }}
          />
        ))}
      </div>

      <DetailedInfo 
        title="Neural Networks - Complete Guide"
        sections={[
          {
            title: "What are Neural Networks?",
            content: "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers that process information using a connectionist approach to computation. The network learns to perform tasks by considering examples, generally without being programmed with task-specific rules."
          },
          {
            title: "Architecture Components",
            content: `**Input Layer**: Receives raw data (images, text, numbers). Each neuron represents one feature of the input data.

**Hidden Layers**: Process information from previous layers. Deep networks have multiple hidden layers, hence "deep learning". Each layer extracts increasingly complex features.

**Output Layer**: Produces final predictions or classifications. Number of neurons depends on the task (e.g., 10 neurons for digit recognition 0-9).

**Weights & Biases**: Each connection has a weight (importance) and each neuron has a bias (threshold). These are the learnable parameters adjusted during training.`
          },
          {
            title: "How Do They Work?",
            content: `**Forward Propagation**:
1. Input data enters the network
2. Each neuron calculates: output = activation(Σ(weight × input) + bias)
3. Data flows through all layers
4. Final layer produces prediction

**Activation Functions**:
- **ReLU**: f(x) = max(0, x) - Most common, prevents vanishing gradients
- **Sigmoid**: f(x) = 1/(1+e^-x) - Outputs between 0 and 1
- **Tanh**: f(x) = (e^x - e^-x)/(e^x + e^-x) - Outputs between -1 and 1
- **Softmax**: Converts outputs to probabilities for classification

**Backpropagation**:
1. Compare prediction with actual output (calculate loss)
2. Use chain rule to compute gradients
3. Update weights using gradient descent: weight_new = weight_old - learning_rate × gradient
4. Repeat for all training examples`
          },
          {
            title: "Types of Neural Networks",
            content: `**Feedforward Neural Networks (FNN)**: Simple architecture where information flows in one direction. Used for basic classification and regression.

**Convolutional Neural Networks (CNN)**: Specialized for image processing. Uses convolutional layers to detect patterns like edges, textures, and objects. Applications: image recognition, object detection, medical imaging.

**Recurrent Neural Networks (RNN)**: Process sequential data with memory of previous inputs. Used for time series, natural language processing, speech recognition.

**Long Short-Term Memory (LSTM)**: Advanced RNN that handles long-term dependencies. Solves vanishing gradient problem in standard RNNs.

**Transformer Networks**: Attention-based architecture that processes entire sequences simultaneously. Powers modern LLMs like GPT, BERT, and Claude.`
          },
          {
            title: "Training Process",
            content: `**Step 1 - Initialize**: Random weights and biases

**Step 2 - Forward Pass**: Input data flows through network to generate predictions

**Step 3 - Calculate Loss**: Measure prediction error using loss functions:
- Mean Squared Error (MSE) for regression: L = (1/n)Σ(predicted - actual)²
- Cross-Entropy Loss for classification: L = -Σ(actual × log(predicted))

**Step 4 - Backward Pass**: Calculate gradients of loss with respect to each weight

**Step 5 - Update Weights**: Use optimization algorithms:
- **SGD**: Simple gradient descent
- **Adam**: Adaptive learning rates (most popular)
- **RMSprop**: Good for recurrent networks

**Step 6 - Repeat**: Iterate through entire dataset multiple times (epochs) until convergence`
          },
          {
            title: "Real-World Applications",
            content: `**Computer Vision**: Image classification (identifying objects), facial recognition, medical diagnosis from X-rays/MRIs, autonomous vehicles, quality control in manufacturing.

**Natural Language Processing**: Language translation, sentiment analysis, chatbots, text generation, question answering, document summarization.

**Healthcare**: Disease diagnosis, drug discovery, protein structure prediction, personalized treatment recommendations, medical image analysis.

**Finance**: Fraud detection, stock price prediction, credit scoring, algorithmic trading, risk assessment.

**Gaming & Entertainment**: Game AI, content recommendation (Netflix, YouTube), deepfakes, music generation, art creation.

**Science**: Weather forecasting, particle physics, astronomy, climate modeling, materials science.`
          },
          {
            title: "Challenges & Solutions",
            content: `**Overfitting**: Model memorizes training data but fails on new data.
Solutions: Dropout (randomly disable neurons), L1/L2 regularization, early stopping, data augmentation.

**Vanishing Gradients**: Gradients become too small in deep networks.
Solutions: ReLU activation, batch normalization, residual connections (skip connections).

**Exploding Gradients**: Gradients become too large.
Solutions: Gradient clipping, careful weight initialization, batch normalization.

**Computational Cost**: Training requires significant resources.
Solutions: GPU/TPU acceleration, model compression, transfer learning, efficient architectures.

**Data Requirements**: Need large labeled datasets.
Solutions: Transfer learning, data augmentation, semi-supervised learning, synthetic data generation.`
          },
          {
            title: "Getting Started - Python Example",
            content: `# Simple Neural Network with PyTorch
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)  # Input: 28x28 image
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)    # Output: 10 classes
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Create model, loss, optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')`
          },
          {
            title: "Key Hyperparameters",
            content: `**Learning Rate**: Controls step size during optimization. Too high = unstable, too low = slow convergence. Typical: 0.001 - 0.1

**Batch Size**: Number of samples processed before updating weights. Typical: 32, 64, 128, 256. Larger = more stable but slower.

**Epochs**: Number of complete passes through training data. Typical: 10-100+ depending on dataset size.

**Number of Layers**: Depth of network. Deeper = more complex patterns but harder to train. Start small, increase as needed.

**Neurons per Layer**: Width of network. More neurons = more capacity but more computation. Often decreases in deeper layers.

**Dropout Rate**: Probability of disabling neurons during training. Typical: 0.2 - 0.5 for preventing overfitting.

**Activation Functions**: Choice affects learning dynamics. ReLU for most cases, Sigmoid/Tanh for gates, Softmax for outputs.`
          },
          {
            title: "Resources to Learn More",
            content: `**Online Courses**:
- Andrew Ng's Machine Learning (Coursera)
- Fast.ai Practical Deep Learning
- Deep Learning Specialization (Coursera)

**Books**:
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Neural Networks and Deep Learning" by Michael Nielsen
- "Hands-On Machine Learning" by Aurélien Géron

**Frameworks**:
- PyTorch (research-friendly, flexible)
- TensorFlow/Keras (production-ready, easy API)
- JAX (high-performance, functional)

**Practice Platforms**:
- Kaggle (competitions, datasets)
- Google Colab (free GPU access)
- Papers with Code (implementation examples)`
          }
        ]}
      />
    </div>
  );
};

// Reinforcement Learning Demo (keeping original)
const ReinforcementDemo = ({ isPlaying, speed }) => {
  const [agentPos, setAgentPos] = useState({ x: 0, y: 0 });
  const [score, setScore] = useState(0);
  const [steps, setSteps] = useState(0);
  const goalPos = { x: 4, y: 4 };

  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setAgentPos(prev => {
        const dx = goalPos.x - prev.x;
        const dy = goalPos.y - prev.y;
        
        let newX = prev.x;
        let newY = prev.y;

        if (Math.abs(dx) > Math.abs(dy)) {
          newX = prev.x + (dx > 0 ? 1 : -1);
        } else if (dy !== 0) {
          newY = prev.y + (dy > 0 ? 1 : -1);
        }

        const distance = Math.abs(goalPos.x - newX) + Math.abs(goalPos.y - newY);
        const reward = distance === 0 ? 100 : -1;
        
        setScore(s => s + reward);
        setSteps(st => st + 1);

        if (distance === 0) {
          setTimeout(() => {
            setAgentPos({ x: 0, y: 0 });
          }, 1000);
        }

        return { x: newX, y: newY };
      });
    }, 500 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, speed]);

  return (
    <div className="p-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Reinforcement Learning Agent</h2>
        <p className="text-gray-300">An agent learns to navigate to a goal by receiving rewards (+100) or penalties (-1) for its actions.</p>
      </div>

      <div className="flex gap-8 items-start">
        <div className="flex-1">
          <div className="grid grid-cols-5 gap-2 bg-white/5 p-4 rounded-xl">
            {Array.from({ length: 5 }).map((_, y) => (
              Array.from({ length: 5 }).map((_, x) => (
                <div
                  key={`${x}-${y}`}
                  className={`w-16 h-16 rounded-lg flex items-center justify-center text-2xl transition-all ${
                    x === agentPos.x && y === agentPos.y
                      ? 'bg-gradient-to-br from-purple-500 to-pink-500 scale-110 shadow-xl'
                      : x === goalPos.x && y === goalPos.y
                      ? 'bg-gradient-to-br from-green-500 to-emerald-500'
                      : 'bg-white/10'
                  }`}
                >
                  {x === agentPos.x && y === agentPos.y && '🤖'}
                  {x === goalPos.x && y === goalPos.y && '🎯'}
                </div>
              ))
            ))}
          </div>
        </div>

        <div className="w-64 space-y-4">
          <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-4 rounded-xl">
            <div className="text-sm text-purple-100">Total Score</div>
            <div className="text-3xl font-bold">{score}</div>
          </div>
          <div className="bg-white/10 p-4 rounded-xl">
            <div className="text-sm text-gray-300">Steps Taken</div>
            <div className="text-2xl font-bold">{steps}</div>
          </div>
        </div>
      </div>

      <DetailedInfo 
        title="Reinforcement Learning - Complete Guide"
        sections={[
          {
            title: "What is Reinforcement Learning?",
            content: "Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. Unlike supervised learning (which uses labeled data) or unsupervised learning (which finds patterns), RL learns through trial and error, receiving rewards or penalties for its actions. The goal is to learn a policy that maximizes cumulative reward over time."
          },
          {
            title: "Core Components",
            content: `**Agent**: The learner/decision maker that takes actions.

**Environment**: The world the agent interacts with. It provides feedback based on agent's actions.

**State (S)**: Current situation/configuration of the environment. Complete information needed for decision-making.

**Action (A)**: Choices available to the agent. Can be discrete (move left/right) or continuous (steering angle).

**Reward (R)**: Immediate feedback signal. Positive for good actions, negative for bad ones. Defines the goal.

**Policy (π)**: Strategy that maps states to actions. Can be deterministic π(s)=a or stochastic π(a|s).

**Value Function (V)**: Expected cumulative reward from a state. V(s) = E[R_t+1 + γR_t+2 + γ²R_t+3 + ...]

**Q-Function**: Expected reward for taking action 'a' in state 's'. Q(s,a) guides action selection.

**Discount Factor (γ)**: Value between 0-1 that prioritizes immediate vs future rewards. γ=0 is myopic, γ→1 is far-sighted.`
          },
          {
            title: "The RL Loop",
            content: `1. **Observe** current state S_t
2. **Select** action A_t based on policy π
3. **Execute** action in environment
4. **Receive** reward R_t+1 and next state S_t+1
5. **Update** policy/value function based on experience
6. **Repeat** until task completion or termination

**Markov Decision Process (MDP)**:
RL problems are formalized as MDPs with:
- Set of states S
- Set of actions A
- Transition probabilities P(s'|s,a)
- Reward function R(s,a,s')
- Discount factor γ

**Bellman Equation** (fundamental to RL):
V(s) = max_a [R(s,a) + γ Σ P(s'|s,a)V(s')]
Q(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q(s',a')`
          },
          {
            title: "Major RL Algorithms",
            content: `**Q-Learning** (Model-Free, Off-Policy):
- Learns Q(s,a) values directly
- Update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- Explores using ε-greedy: random action with probability ε
- Guaranteed convergence to optimal policy
- Used in: Atari games, robotics

**SARSA** (Model-Free, On-Policy):
- Similar to Q-learning but updates based on actual next action
- Update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- More conservative than Q-learning
- Better for safety-critical applications

**Deep Q-Network (DQN)**:
- Uses neural network to approximate Q-function
- Experience Replay: stores transitions, samples randomly for training
- Target Network: separate network for stable Q-value targets
- Achieved human-level performance on Atari games

**Policy Gradient Methods**:
- Directly optimize policy π_θ(a|s)
- REINFORCE algorithm: ∇J(θ) = E[∇log π_θ(a|s) Q(s,a)]
- Works with continuous action spaces
- Used in robotics, game playing

**Actor-Critic**:
- Actor: learns policy (what to do)
- Critic: learns value function (how good is current state)
- Combines benefits of value-based and policy-based methods
- Examples: A3C, PPO, SAC

**Proximal Policy Optimization (PPO)**:
- State-of-the-art policy gradient method
- Clips updates to prevent drastic policy changes
- Simple, stable, effective
- Used by OpenAI, DeepMind

**Trust Region Policy Optimization (TRPO)**:
- Guarantees monotonic improvement
- Constrains policy updates to "trust region"
- Computationally expensive but very stable`
          },
          {
            title: "Exploration vs Exploitation",
            content: `**The Dilemma**: Should agent try new actions (explore) or use best known actions (exploit)?

**ε-Greedy Strategy**:
- With probability ε: choose random action (explore)
- With probability 1-ε: choose best action (exploit)
- Decay ε over time: ε = ε_initial × decay_rate^episode

**Boltzmann Exploration**:
- Sample actions based on Q-values with temperature T
- P(a|s) = exp(Q(s,a)/T) / Σ exp(Q(s,a')/T)
- High T = more exploration, Low T = more exploitation

**Upper Confidence Bound (UCB)**:
- Balance exploration and exploitation mathematically
- Select action maximizing: Q(s,a) + c√(ln(t)/N(s,a))
- Explores actions with high uncertainty

**Curiosity-Driven Exploration**:
- Add intrinsic reward for novel states
- Encourages agent to explore unknown areas
- Used in sparse reward environments

**Thompson Sampling**:
- Bayesian approach to exploration
- Sample from posterior distribution of Q-values
- Naturally balances exploration and exploitation`
          },
          {
            title: "Real-World Applications",
            content: `**Robotics**:
- Robot manipulation (grasping, assembly)
- Locomotion (walking, running robots)
- Autonomous navigation
- Multi-robot coordination
- Warehouse automation

**Game Playing**:
- AlphaGo/AlphaZero (defeated world champions)
- Dota 2 bots (OpenAI Five)
- StarCraft II agents (AlphaStar)
- Poker bots (Pluribus)
- Video game AI

**Autonomous Vehicles**:
- Path planning and navigation
- Traffic signal control
- Adaptive cruise control
- Parking assistance
- Fleet management

**Recommendation Systems**:
- Personalized content (Netflix, YouTube)
- Ad placement optimization
- News feed curation
- E-commerce recommendations
- Dynamic pricing

**Finance**:
- Algorithmic trading
- Portfolio optimization
- Risk management
- Option pricing
- Market making

**Healthcare**:
- Treatment optimization
- Drug dosage control
- Dynamic treatment regimes
- Clinical trial design
- Resource allocation

**Energy & Resources**:
- Smart grid management
- HVAC optimization
- Battery charging strategies
- Traffic light control
- Supply chain optimization`
          },
          {
            title: "Challenges in RL",
            content: `**Sample Efficiency**:
Problem: Requires millions of interactions to learn
Solutions: Transfer learning, model-based RL, meta-learning, demonstration learning

**Sparse Rewards**:
Problem: Reward signal is rare, hard to learn
Solutions: Reward shaping, hindsight experience replay, curiosity-driven exploration

**Credit Assignment**:
Problem: Which past actions led to current reward?
Solutions: Temporal difference learning, eligibility traces, attention mechanisms

**Partial Observability**:
Problem: Agent doesn't see complete state
Solutions: Recurrent networks (LSTM), belief states, history window

**Continuous Action Spaces**:
Problem: Infinite actions to consider
Solutions: Policy gradient methods, deterministic policy gradient, continuous control algorithms

**Safety & Constraints**:
Problem: Dangerous exploration in real world
Solutions: Safe exploration, constrained RL, sim-to-real transfer, human oversight

**Non-Stationarity**:
Problem: Environment changes over time
Solutions: Online learning, adaptive policies, meta-RL

**Multi-Agent Interactions**:
Problem: Other agents change environment dynamics
Solutions: Multi-agent RL, game theory, communication protocols`
          },
          {
            title: "Implementation Example",
            content: `# Q-Learning Implementation in Python
import numpy as np
import gym

class QLearningAgent:
    def __init__(self, states, actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((states, actions))
        self.lr = lr          # Learning rate
        self.gamma = gamma    # Discount factor
        self.epsilon = epsilon # Exploration rate
        
    def get_action(self, state):
        # ε-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        # Q-Learning update rule
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

# Training loop
env = gym.make('FrozenLake-v1')
agent = QLearningAgent(env.observation_space.n, env.action_space.n)

for episode in range(10000):
    state = env.reset()[0]
    total_reward = 0
    
    while True:
        action = agent.get_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        
        if done or truncated:
            break
    
    if episode % 1000 == 0:
        print(f'Episode {episode}, Total Reward: {total_reward}')`
          },
          {
            title: "Advanced Topics",
            content: `**Hierarchical RL**:
Learn at multiple temporal scales. High-level controller sets goals, low-level controller achieves them. Enables complex, long-horizon tasks.

**Multi-Agent RL (MARL)**:
Multiple agents learning simultaneously. Cooperation, competition, or mixed scenarios. Applications: autonomous vehicles, multi-robot systems.

**Inverse RL**:
Learn reward function from expert demonstrations. "Why is the expert doing this?" Used for imitation learning and apprenticeship learning.

**Model-Based RL**:
Learn model of environment dynamics. Plan using the model. More sample-efficient but model errors compound.

**Meta-RL**:
Learn to learn. Train on distribution of tasks. Quickly adapt to new tasks. Few-shot learning in RL.

**Offline/Batch RL**:
Learn from fixed dataset without environment interaction. Important for real-world applications where online interaction is expensive or dangerous.

**Multi-Objective RL**:
Optimize multiple, possibly conflicting objectives. Find Pareto-optimal policies. Used in resource management, robotics.`
          },
          {
            title: "Resources & Tools",
            content: `**Libraries & Frameworks**:
- OpenAI Gym: Standard RL benchmarks
- Stable-Baselines3: PyTorch implementations
- RLlib (Ray): Scalable RL library
- TF-Agents: TensorFlow-based RL
- Dopamine: Research framework by Google

**Learning Resources**:
- Sutton & Barto "Reinforcement Learning: An Introduction"
- David Silver's RL Course (DeepMind)
- Spinning Up in Deep RL (OpenAI)
- CS285 Berkeley Deep RL course

**Simulation Environments**:
- MuJoCo: Physics simulation for robotics
- Unity ML-Agents: Game-based RL
- Isaac Gym: GPU-accelerated robot simulation
- Atari: Classic game benchmarks
- DeepMind Control Suite: Continuous control tasks`
          }
        ]}
      />
    </div>
  );
};

// Agentic AI Demo (keeping original)
const AgenticAIDemo = ({ isPlaying, speed }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [thoughts, setThoughts] = useState([]);

  const steps = [
    { phase: 'Perceive', desc: 'Analyze environment & goals', color: 'from-cyan-500 to-blue-500', icon: '👁️' },
    { phase: 'Reason', desc: 'Plan sequence of actions', color: 'from-purple-500 to-pink-500', icon: '🧠' },
    { phase: 'Act', desc: 'Execute chosen action', color: 'from-green-500 to-emerald-500', icon: '⚡' },
    { phase: 'Learn', desc: 'Update knowledge from feedback', color: 'from-orange-500 to-red-500', icon: '📚' }
  ];

  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentStep(prev => (prev + 1) % steps.length);
      setThoughts(prev => [...prev.slice(-3), {
        id: Date.now(),
        text: steps[(currentStep + 1) % steps.length].desc,
        phase: steps[(currentStep + 1) % steps.length].phase
      }]);
    }, 2000 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, speed, currentStep]);

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
        sections={[
          {
            title: "What is Agentic AI?",
            content: "Agentic AI refers to artificial intelligence systems that act as autonomous agents - capable of perceiving their environment, making decisions, taking actions, and learning from outcomes to achieve specific goals without constant human supervision. Unlike traditional AI that simply responds to queries, agentic AI proactively plans, executes multi-step tasks, uses tools, and adapts its behavior based on feedback. It represents a shift from passive AI assistants to active AI collaborators."
          },
          {
            title: "Key Characteristics",
            content: `**Autonomy**: Operates independently with minimal human intervention. Makes decisions without waiting for explicit instructions.

**Goal-Oriented**: Driven by objectives rather than reactive responses. Plans sequences of actions to achieve desired outcomes.

**Proactivity**: Initiates actions and anticipates needs rather than only responding to prompts. Takes initiative in problem-solving.

**Adaptability**: Learns from experience and adjusts behavior. Handles unexpected situations and recovers from failures.

**Tool Use**: Leverages external resources (APIs, databases, code execution, web search) to extend capabilities beyond base model.

**Memory & State**: Maintains context across interactions. Remembers past actions, decisions, and their outcomes.

**Multi-Step Reasoning**: Breaks complex problems into subtasks. Executes plans with multiple intermediate steps.

**Self-Reflection**: Evaluates own performance, identifies errors, and self-corrects. Meta-cognitive awareness of capabilities and limitations.`
          },
          {
            title: "Core Components",
            content: `**Perception Module**:
- Processes input from environment (text, images, structured data)
- Identifies relevant information and context
- Monitors state changes and feedback signals
- Example: Reading emails, analyzing documents, observing API responses

**Reasoning Engine**:
- Plans sequences of actions (chain-of-thought, tree search)
- Evaluates options and makes decisions
- Decomposes complex goals into subtasks
- Handles uncertainty and ambiguity
- Often powered by Large Language Models (LLMs)

**Action Execution**:
- Interfaces with external tools and APIs
- Executes code, makes API calls, writes files
- Manipulates environment based on decisions
- Handles errors and retries

**Memory Systems**:
- **Working Memory**: Current context and immediate task state
- **Episodic Memory**: Past experiences and interaction history
- **Semantic Memory**: General knowledge and learned facts
- **Procedural Memory**: How to perform specific tasks

**Learning & Adaptation**:
- Updates based on success/failure feedback
- Refines strategies over time
- Learns from demonstrations and corrections
- Transfer learning across similar tasks

**Control & Monitoring**:
- Tracks progress toward goals
- Detects when stuck or failing
- Triggers re-planning when needed
- Manages computational resources`
          },
          {
            title: "Agentic AI Frameworks",
            content: `**ReAct (Reason + Act)**:
Alternates between reasoning and acting. Each step includes:
1. Thought: What should I do next?
2. Action: Execute chosen action
3. Observation: What happened?
4. Repeat until goal achieved

Example:
Thought: I need to find current weather
Action: search("weather New York today")
Observation: [search results showing 72°F, sunny]
Thought: Now I have the answer
Action: respond("It's 72°F and sunny in New York")

**AutoGPT Pattern**:
Autonomous goal achievement through:
- Goal decomposition into subtasks
- Iterative task execution
- Self-critique and improvement
- Memory of past actions
- Continuous operation until success

**BabyAGI**:
Task-driven autonomous agent:
1. Pull task from priority queue
2. Execute task using LLM
3. Enrich results and store in memory
4. Create new tasks based on results
5. Reprioritize task list

**LangChain Agents**:
Flexible framework with:
- Agent types (zero-shot, conversational, structured)
- Tool integration (search, calculators, APIs)
- Memory management
- Chain composition
- Prompt engineering utilities

**CrewAI**:
Multi-agent orchestration:
- Define roles (researcher, writer, analyst)
- Assign tasks to specialized agents
- Coordinate collaboration
- Aggregate results

**Semantic Kernel**:
Microsoft's framework for:
- Skill composition
- Planner integration
- Memory management
- Plugin architecture`
          },
          {
            title: "Agent Architectures",
            content: `**Single-Agent Systems**:
One agent handles entire task. Simple but limited by single perspective and capabilities.

**Multi-Agent Systems**:
Multiple specialized agents collaborate:
- **Hierarchical**: Manager delegates to specialist agents
- **Peer-to-Peer**: Agents communicate and coordinate directly
- **Pipeline**: Sequential processing by different agents
- **Debate**: Multiple agents discuss and reach consensus

**Agent Roles** (in multi-agent systems):
- **Planner**: Creates task breakdown and execution strategy
- **Executor**: Performs actions and uses tools
- **Researcher**: Gathers information from various sources
- **Critic**: Evaluates outputs and provides feedback
- **Synthesizer**: Combines results from multiple agents

**Cognitive Architectures**:
- **ACT-R**: Models human cognitive processes
- **SOAR**: General intelligence architecture
- **CLARION**: Hybrid symbolic-connectionist system
- **Attention-based**: Focus on relevant information dynamically`
          },
          {
            title: "Tool Use & Integration",
            content: `**Common Tools**:
- **Web Search**: Access current information beyond training data
- **Code Execution**: Run Python, JavaScript for computations
- **Database Access**: Query structured data (SQL, NoSQL)
- **API Calls**: Interact with external services (weather, stocks, etc.)
- **File Operations**: Read/write documents, process data
- **Image Generation**: Create visuals (DALL-E, Stable Diffusion)
- **Web Scraping**: Extract information from websites

**Tool Selection Strategies**:
- Rule-based: Predefined conditions trigger specific tools
- Learning-based: Agent learns which tools work best
- LLM-powered: Language model decides tool usage
- Hierarchical: Tool orchestration layer manages access

**Error Handling**:
- Retry with different parameters
- Fallback to alternative tools
- Request human intervention
- Graceful degradation

**Tool Chaining**:
Use output of one tool as input to another:
search → extract_text → summarize → translate → save_file`
          },
          {
            title: "Real-World Applications",
            content: `**Personal Assistants**:
- Email management and response drafting
- Calendar scheduling and coordination
- Task prioritization and reminders
- Travel planning and booking
- Research and information gathering

**Software Development**:
- Code generation and debugging
- Test case creation
- Documentation writing
- Code review and refactoring
- Dependency management
- Example: GitHub Copilot, Cursor, Replit Agent

**Business Operations**:
- Customer service automation
- Data analysis and reporting
- Invoice processing and accounting
- Supply chain optimization
- Market research and competitor analysis

**Scientific Research**:
- Literature review and synthesis
- Hypothesis generation
- Experiment design
- Data analysis and visualization
- Paper writing assistance

**Content Creation**:
- Blog post writing with research
- Social media management
- Video script generation
- Marketing copy optimization
- SEO content strategy

**Healthcare**:
- Patient triage and assessment
- Medical literature search
- Treatment plan suggestions
- Administrative task automation
- Drug interaction checking`
          },
          {
            title: "Challenges & Limitations",
            content: `**Reliability**:
- Hallucinations and factual errors
- Inconsistent reasoning across attempts
- Difficulty with complex multi-step tasks
- Solutions: Verification steps, human-in-the-loop, confidence scoring

**Cost & Latency**:
- Multiple LLM calls expensive
- Real-time response requirements challenging
- Token limit constraints
- Solutions: Caching, model distillation, efficient architectures

**Safety & Control**:
- Unintended actions with real consequences
- Difficulty predicting agent behavior
- Potential for misuse
- Solutions: Sandboxing, approval workflows, monitoring, kill switches

**Evaluation**:
- Hard to benchmark autonomous behavior
- Success metrics ambiguous
- Difficulty comparing different approaches
- Solutions: Task-specific metrics, human evaluation, simulation environments

**Alignment**:
- Understanding true user intent
- Balancing multiple objectives
- Value alignment with human preferences
- Solutions: Reinforcement learning from human feedback (RLHF), constitutional AI

**Context Management**:
- Limited context windows
- Information prioritization
- Handling long-running tasks
- Solutions: Hierarchical memory, summarization, retrieval augmentation`
          },
          {
            title: "Implementation Example",
            content: `# Simple Agentic AI with ReAct Pattern
from anthropic import Anthropic
import json

class SimpleAgent:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.tools = {
            'search': self.search_tool,
            'calculate': self.calculate_tool,
            'get_weather': self.weather_tool
        }
        self.memory = []
    
    def search_tool(self, query):
        # Simulated search
        return f"Search results for: {query}"
    
    def calculate_tool(self, expression):
        try:
            return str(eval(expression))
        except:
            return "Calculation error"
    
    def weather_tool(self, location):
        return f"Weather in {location}: 72°F, Sunny"
    
    def run(self, goal, max_steps=10):
        self.memory.append({"role": "user", "content": goal})
        
        for step in range(max_steps):
            # Reasoning step
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system="You are an autonomous agent. Use ReAct pattern: Thought, Action, Observation.",
                messages=self.memory
            )
            
            content = response.content[0].text
            self.memory.append({"role": "assistant", "content": content})
            
            # Check if task complete
            if "TASK COMPLETE" in content:
                return content
            
            # Parse and execute action
            if "Action:" in content:
                action_line = [l for l in content.split('\n') if 'Action:' in l][0]
                tool_name = action_line.split(':')[1].strip().split('(')[0]
                
                if tool_name in self.tools:
                    # Execute tool
                    result = self.tools[tool_name]()
                    observation = f"Observation: {result}"
                    self.memory.append({"role": "user", "content": observation})
        
        return "Max steps reached"

# Usage
agent = SimpleAgent(api_key="your-key")
result = agent.run("What's the weather in New York and how does it compare to average?")`
          },
          {
            title: "Future Directions",
            content: `**Enhanced Reasoning**:
- System 2 thinking (slow, deliberate reasoning)
- Causal reasoning and counterfactual thinking
- Abstract reasoning and analogy
- Mathematical and symbolic reasoning

**Improved Planning**:
- Long-horizon planning (days/weeks ahead)
- Uncertainty-aware planning
- Multi-objective optimization
- Dynamic replanning

**Better Learning**:
- Few-shot task learning
- Continuous learning from interactions
- Transfer across domains
- Meta-learning (learning to learn)

**Human-AI Collaboration**:
- Natural language instruction following
- Intent clarification and confirmation
- Graceful handling of ambiguity
- Shared mental models

**Specialized Agents**:
- Domain-specific expertise (medical, legal, engineering)
- Vertical integration with industry tools
- Compliance and regulatory awareness

**Embodied Agents**:
- Physical robot control
- Sensory-motor integration
- Real-world interaction
- Spatial reasoning

**Ethical & Responsible AI**:
- Transparent decision-making
- Explainable actions
- Fairness and bias mitigation
- Privacy preservation
- Accountability mechanisms`
          },
          {
            title: "Best Practices",
            content: `**Design Principles**:
- Start simple, add complexity gradually
- Clear goal specification
- Explicit success/failure criteria
- Fail-safe defaults
- Human oversight for critical actions

**Prompt Engineering**:
- Detailed system prompts with examples
- Structured output formats (JSON, XML)
- Chain-of-thought prompting
- Role specification
- Constraint definition

**Testing & Validation**:
- Unit test individual components
- Integration test tool chains
- Adversarial testing (edge cases)
- Human evaluation
- A/B testing different approaches

**Monitoring & Logging**:
- Track all actions and decisions
- Log reasoning traces
- Monitor costs and latency
- Detect anomalous behavior
- Version control for prompts

**Iterative Development**:
- Build minimum viable agent
- Gather feedback
- Identify failure modes
- Refine and expand capabilities
- Continuous improvement cycle`
          },
          {
            title: "Learning Resources",
            content: `**Courses & Tutorials**:
- DeepLearning.AI: AI Agents in LangGraph
- Andrew Ng: Building Agentic RAG
- LangChain Agents Documentation
- Anthropic: Building Effective Agents

**Research Papers**:
- "ReAct: Synergizing Reasoning and Acting in Language Models"
- "Toolformer: Language Models Can Teach Themselves to Use Tools"
- "AutoGPT: An Autonomous GPT-4 Experiment"
- "Generative Agents: Interactive Simulacra of Human Behavior"

**Frameworks & Tools**:
- LangChain / LangGraph
- AutoGen (Microsoft)
- Semantic Kernel
- CrewAI
- Agent Protocol

**Communities**:
- LangChain Discord
- AutoGPT Community
- AI Agent Development subreddits
- Anthropic Developer Forums`
          }
        ]}
      />
    </div>
  );
};

// Multi-Agent Systems Demo (keeping original)
const MultiAgentDemo = ({ isPlaying, speed }) => {
  const [agents, setAgents] = useState([
    { id: 1, x: 20, y: 30, role: 'Explorer', color: 'from-blue-500 to-cyan-500', target: null },
    { id: 2, x: 80, y: 30, role: 'Analyzer', color: 'from-purple-500 to-pink-500', target: null },
    { id: 3, x: 50, y: 70, role: 'Executor', color: 'from-green-500 to-emerald-500', target: null }
  ]);
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setAgents(prev => prev.map(agent => ({
        ...agent,
        x: agent.x + (Math.random() - 0.5) * 10,
        y: agent.y + (Math.random() - 0.5) * 10
      })));

      const newMsg = {
        id: Date.now(),
        from: agents[Math.floor(Math.random() * agents.length)].role,
        to: agents[Math.floor(Math.random() * agents.length)].role,
        text: ['Share data', 'Task complete', 'Need assistance', 'Coordinating'][Math.floor(Math.random() * 4)]
      };
      setMessages(prev => [...prev.slice(-4), newMsg]);
    }, 1500 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, speed]);

  return (
    <div className="p-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Multi-Agent Collaboration</h2>
        <p className="text-gray-300">Multiple AI agents work together, communicating and coordinating to solve complex tasks beyond single-agent capabilities.</p>
      </div>

      <div className="flex gap-8">
        <div className="flex-1 bg-white/5 rounded-xl p-6 relative h-96">
          {agents.map(agent => (
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
              {messages.map(msg => (
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
        sections={[
          {
            title: "What are Multi-Agent Systems?",
            content: "Multi-Agent Systems (MAS) are computational systems where multiple autonomous agents interact and coordinate to achieve individual or collective goals. Each agent has its own knowledge, capabilities, and objectives, and they communicate and collaborate to solve problems that are beyond the capacity of individual agents. MAS enables distributed problem-solving, parallel processing, and emergent intelligent behavior from relatively simple individual agents."
          },
          {
            title: "Core Concepts",
            content: `**Agent**: Autonomous entity with:
- Perception: Observes environment
- Decision-making: Chooses actions
- Action: Affects environment
- Communication: Exchanges information with other agents
- Learning: Improves over time

**Environment**:
- **Accessible**: Agents can perceive relevant state
- **Deterministic/Stochastic**: Predictable vs probabilistic outcomes
- **Episodic/Sequential**: Independent vs connected actions
- **Static/Dynamic**: Changes while agent deliberates
- **Discrete/Continuous**: Finite vs infinite states

**Interaction Types**:
- **Cooperative**: Agents share common goals
- **Competitive**: Agents have conflicting goals
- **Mixed**: Some cooperation, some competition
- **Coordination**: Synchronize actions without explicit communication

**Communication**:
- **Direct**: Message passing between agents
- **Indirect**: Through environment (stigmergy)
- **Broadcast**: One-to-many communication
- **Protocol-based**: Structured interaction patterns

**Organization**:
- **Flat**: All agents equal status
- **Hierarchical**: Manager-worker relationships
- **Market-based**: Economic coordination
- **Social**: Role-based interactions`
          },
          {
            title: "Agent Architectures",
            content: `**Reactive Agents**:
- Simple stimulus-response behavior
- No internal state or planning
- Fast, robust, but limited
- Example: Braitenberg vehicles, subsumption architecture

**Deliberative Agents**:
- Explicit symbolic world model
- Plan actions to achieve goals
- BDI (Belief-Desire-Intention) architecture
- Slower but more flexible

**Hybrid Agents**:
- Combine reactive and deliberative layers
- Reactive for immediate responses
- Deliberative for complex planning
- Example: Three-layer architecture (reactive, executive, deliberative)

**Learning Agents**:
- Adapt behavior based on experience
- Reinforcement learning common
- Policy improvement over time
- Handle unknown environments

**Cognitive Agents**:
- Human-like reasoning
- Mental states (beliefs, desires, intentions, emotions)
- Theory of mind (model other agents)
- Social intelligence`
          },
          {
            title: "Coordination Mechanisms",
            content: `**Negotiation**:
Agents bargain to reach agreement
- **Contract Net Protocol**: Task announcement, bidding, awarding
- **Auctions**: VCG, English, Dutch, sealed-bid
- **Argumentation**: Exchange reasons and justifications
- **Game Theory**: Nash equilibrium, mechanism design

**Planning**:
- **Centralized**: Single planner coordinates all agents
- **Distributed**: Each agent plans independently
- **Partial Global Planning**: Agents share plan fragments
- **HTN Planning**: Hierarchical task network decomposition

**Voting & Consensus**:
- Majority voting for group decisions
- Weighted voting based on expertise
- Consensus protocols (Byzantine agreement)
- Social choice theory

**Blackboard Systems**:
- Shared knowledge space
- Agents read/write to blackboard
- Opportunistic problem solving
- Example: HEARSAY-II speech recognition

**Market Mechanisms**:
- Resource allocation through prices
- Supply-demand equilibrium
- Virtual currencies and budgets
- Combinatorial auctions

**Norms & Conventions**:
- Shared rules and expectations
- Emergent social behavior
- Stigmergy (indirect coordination)
- Traffic rules, queue discipline`
          },
          {
            title: "Communication Protocols",
            content: `**FIPA ACL** (Agent Communication Language):
Standard message structure:
- Performative: Type of communication act
- Sender/Receiver: Agent identifiers
- Content: Message payload
- Ontology: Shared vocabulary
- Protocol: Interaction pattern

**Common Performatives**:
- INFORM: Share information
- REQUEST: Ask for action
- QUERY: Ask question
- PROPOSE: Suggest course of action
- ACCEPT/REJECT: Response to proposals
- CFP (Call for Proposals): Initiate negotiation

**Interaction Protocols**:
- **Request**: Simple request-response
- **Contract Net**: Task allocation through bidding
- **Auction**: Competitive resource allocation
- **Brokering**: Intermediary-mediated interaction
- **Recruiting**: Team formation

**Message Semantics**:
- Speech act theory
- Conversational implicature
- Commitment-based semantics
- Social semantics (obligations, permissions)

**Ontologies**:
- Shared conceptual models
- Domain vocabulary
- Relationships between concepts
- Reasoning support
- Examples: FOAF, SUMO, industry-specific ontologies`
          },
          {
            title: "Multi-Agent Learning",
            content: `**Independent Learners**:
Each agent learns ignoring others
- Simple but limited
- Treats other agents as part of environment
- Non-stationary environment problem

**Joint Action Learning**:
Agents learn to coordinate actions
- Q-learning with joint action space
- Exponential growth in state space
- Scalability challenges

**Opponent Modeling**:
Learn models of other agents' behavior
- Predict actions of others
- Exploit predictable opponents
- Theory of mind reasoning

**Communication Learning**:
Learn when and what to communicate
- Minimize communication overhead
- Emergent communication protocols
- Language grounding

**Cooperative Learning**:
- **Team Q-Learning**: Shared Q-values
- **Distributed Q-Learning**: Local updates, global optimization
- **Actor-Critic**: Policy gradient methods
- **QMIX**: Value decomposition networks
- **MADDPG**: Multi-agent DDPG

**Emergent Behaviors**:
Complex behavior from simple rules
- Flocking (boids algorithm)
- Swarm intelligence
- Collective decision-making
- Self-organization

**Meta-Learning**:
Learn to learn across tasks
- Fast adaptation to new scenarios
- Transfer learning between agents
- Learning communication strategies`
          },
          {
            title: "Real-World Applications",
            content: `**Autonomous Vehicles**:
- Cooperative driving and platooning
- Intersection management without traffic lights
- V2V (vehicle-to-vehicle) communication
- Fleet coordination and routing
- Parking space allocation

**Smart Grids**:
- Distributed energy management
- Load balancing across grid
- Renewable energy integration
- Demand response coordination
- Peer-to-peer energy trading

**Supply Chain**:
- Logistics and warehouse automation
- Multi-company coordination
- Dynamic routing and scheduling
- Inventory management
- Supplier-buyer negotiation

**E-Commerce**:
- Automated trading agents
- Price comparison and monitoring
- Auction participation
- Recommendation systems
- Fraud detection networks

**Gaming & Entertainment**:
- NPC (non-player character) coordination
- Team AI in multiplayer games
- Procedural content generation
- Virtual worlds simulation
- Sports game AI

**Disaster Response**:
- Robot search and rescue teams
- Resource allocation and coordination
- Distributed sensing and mapping
- Communication network establishment
- Task allocation under uncertainty

**Scientific Research**:
- Distributed data analysis
- Collaborative hypothesis testing
- Multi-telescope coordination
- Drug discovery simulations
- Climate modeling

**Social Simulation**:
- Economic modeling
- Epidemic spread simulation
- Traffic pattern analysis
- Urban planning
- Policy evaluation`
          },
          {
            title: "Challenges & Research Areas",
            content: `**Scalability**:
Problem: Performance degrades with many agents
Solutions:
- Hierarchical organization
- Local interaction limitations
- Approximation algorithms
- Divide-and-conquer strategies

**Non-Stationarity**:
Problem: Other agents change behavior
Solutions:
- Opponent modeling
- Robust learning algorithms
- Meta-game analysis
- Adaptive strategies

**Credit Assignment**:
Problem: Which agent contributed to success?
Solutions:
- Difference rewards
- Shapley values
- Counterfactual reasoning
- Local reward signals

**Communication Overhead**:
Problem: Too much communication is expensive
Solutions:
- Learned communication strategies
- Bandwidth constraints
- Information theory approaches
- Selective communication

**Partial Observability**:
Problem: Agents don't see full state
Solutions:
- Decentralized POMDPs
- Belief state maintenance
- Communication for state sharing
- Recurrent neural networks

**Safety & Robustness**:
Problem: Unpredictable emergent behavior
Solutions:
- Formal verification
- Bounded behavior guarantees
- Fault tolerance
- Human oversight

**Heterogeneity**:
Problem: Different agent types and capabilities
Solutions:
- Role assignment
- Capability-aware coordination
- Adaptive team formation
- Standardized interfaces`
          },
          {
            title: "Implementation Example",
            content: `# Multi-Agent System with Mesa Framework
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class CooperativeAgent(Agent):
    def __init__(self, unique_id, model, agent_type):
        super().__init__(unique_id, model)
        self.agent_type = agent_type  # 'explorer', 'analyzer', 'executor'
        self.energy = 100
        self.messages = []
    
    def step(self):
        # Agent behavior based on type
        if self.agent_type == 'explorer':
            self.explore()
        elif self.agent_type == 'analyzer':
            self.analyze()
        else:
            self.execute()
        
        # Share information with nearby agents
        self.communicate()
    
    def explore(self):
        # Move randomly, discover resources
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_pos = self.random.choice(neighbors)
        self.model.grid.move_agent(self, new_pos)
        self.energy -= 1
    
    def analyze(self):
        # Process information from explorers
        nearby = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=2
        )
        for agent in nearby:
            if agent.agent_type == 'explorer':
                self.messages.extend(agent.messages)
    
    def execute(self):
        # Act on analyzed information
        if len(self.messages) > 5:
            # Have enough info to act
            self.energy += 10
            self.messages.clear()
    
    def communicate(self):
        # Broadcast to nearby agents
        nearby = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=1
        )
        message = {
            'sender': self.unique_id,
            'type': self.agent_type,
            'energy': self.energy,
            'position': self.pos
        }
        for agent in nearby:
            agent.messages.append(message)

class MultiAgentModel(Model):
    def __init__(self, n_agents=10, width=20, height=20):
        self.num_agents = n_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        # Create agents
        for i in range(n_agents):
            agent_type = ['explorer', 'analyzer', 'executor'][i % 3]
            agent = CooperativeAgent(i, self, agent_type)
            self.schedule.add(agent)
            
            # Place agent randomly
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
        
        # Data collection
        self.datacollector = DataCollector(
            agent_reporters={"Energy": "energy"}
        )
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Run simulation
model = MultiAgentModel(n_agents=15)
for i in range(100):
    model.step()
    
# Analyze results
agent_energy = model.datacollector.get_agent_vars_dataframe()
print(agent_energy.groupby("AgentID")["Energy"].mean())`
          },
          {
            title: "Advanced Topics",
            content: `**Swarm Intelligence**:
Emergent collective behavior from simple agents
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)
- Bee algorithms
- Fish schooling, bird flocking

**Coalition Formation**:
Agents form groups to achieve goals
- Coalition structure generation
- Stability (core, Nash-stable)
- Payoff distribution (Shapley value)
- Dynamic coalitions

**Argumentation**:
Reason about conflicting information
- Argument graphs
- Attack and support relationships
- Dung's argumentation frameworks
- Dialectical reasoning

**Trust & Reputation**:
Evaluate reliability of other agents
- Direct trust (personal experience)
- Indirect trust (recommendations)
- Trust propagation
- Reputation systems (eBay, Airbnb)

**Normative Systems**:
Social rules governing behavior
- Obligation, permission, prohibition
- Norm emergence and evolution
- Sanction mechanisms
- Legal reasoning

**Organizational Models**:
Structure and roles in MAS
- MOISE+ framework
- Role dependencies
- Organizational constraints
- Dynamic reorganization

**Semantic Web Agents**:
Agents on the web
- Web services composition
- Ontology-based reasoning
- Linked data integration
- Autonomous web applications`
          },
          {
            title: "Frameworks & Tools",
            content: `**Agent Platforms**:
- **JADE**: Java Agent DEvelopment Framework
- **Jason**: AgentSpeak interpreter
- **SPADE**: Smart Python Agent Development Environment
- **NetLogo**: Agent-based modeling environment
- **Mesa**: Python agent-based modeling
- **Repast**: Recursive Porous Agent Simulation Toolkit

**Communication**:
- FIPA ACL (Agent Communication Language)
- KQML (Knowledge Query Manipulation Language)
- HTTP/REST APIs
- Message queues (RabbitMQ, Kafka)

**LLM-based Multi-Agent**:
- **AutoGen** (Microsoft): Conversational agents
- **CrewAI**: Role-based agent teams
- **LangGraph**: Multi-agent workflows
- **AgentVerse**: Large-scale agent simulation

**Simulation Environments**:
- **StarCraft II**: Real-time strategy
- **Hanabi**: Cooperative card game
- **Overcooked**: Coordination challenge
- **Multi-Agent Particle Environment**

**Benchmarks**:
- SMAC (StarCraft Multi-Agent Challenge)
- MPE (Multi-Agent Particle Environment)
- PettingZoo: Multi-agent RL library
- Google Research Football`
          },
          {
            title: "Learning Resources",
            content: `**Books**:
- "Multiagent Systems" by Wooldridge
- "An Introduction to MultiAgent Systems" by Wooldridge
- "Multiagent Systems: Algorithmic, Game-Theoretic" by Shoham & Leyton-Brown

**Courses**:
- Stanford CS 269I: Multi-Agent Systems
- MIT 6.S191: Multi-Agent Reinforcement Learning
- Coursera: Multi-Agent Systems

**Research Venues**:
- AAMAS (Autonomous Agents and Multi-Agent Systems)
- IJCAI (Multi-Agent Systems track)
- NeurIPS (Multi-Agent RL workshop)

**Papers**:
- "Value-Decomposition Networks For Cooperative Multi-Agent Learning"
- "QMIX: Monotonic Value Function Factorisation"
- "The Surprising Effectiveness of PPO in Cooperative"
- "Emergent Tool Use from Multi-Agent Interaction"

**Online Resources**:
- AgentLink community
- Multi-Agent Systems Lab websites
- OpenAI multi-agent research
- DeepMind multi-agent papers`
          }
        ]}
      />
    </div>
  );
};

// Detailed Information Component
const DetailedInfo = ({ title, sections }) => {
  const [expandedSections, setExpandedSections] = useState(new Set([0]));

  const toggleSection = (index) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSections(newExpanded);
  };

  const expandAll = () => {
    setExpandedSections(new Set(sections.map((_, i) => i)));
  };

  const collapseAll = () => {
    setExpandedSections(new Set());
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
                  <div className="text-gray-300 whitespace-pre-wrap leading-relaxed">
                    {section.content.split('\n').map((paragraph, pIndex) => {
                      // Handle bold text with **
                      if (paragraph.includes('**')) {
                        const parts = paragraph.split('**');
                        return (
                          <p key={pIndex} className="mb-3">
                            {parts.map((part, partIndex) =>
                              partIndex % 2 === 1 ? (
                                <strong key={partIndex} className="text-purple-300 font-semibold">
                                  {part}
                                </strong>
                              ) : (
                                <span key={partIndex}>{part}</span>
                              )
                            )}
                          </p>
                        );
                      }
                      
                      // Code blocks
                      if (paragraph.trim().startsWith('#') || paragraph.trim().startsWith('import')) {
                        return (
                          <pre key={pIndex} className="bg-black/30 p-4 rounded-lg overflow-x-auto mb-3">
                            <code className="text-green-300 text-sm font-mono">
                              {paragraph}
                            </code>
                          </pre>
                        );
                      }
                      
                      // Regular paragraphs
                      if (paragraph.trim()) {
                        return (
                          <p key={pIndex} className="mb-3">
                            {paragraph}
                          </p>
                        );
                      }
                      return null;
                    })}
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

export default AIConceptsDemo;
