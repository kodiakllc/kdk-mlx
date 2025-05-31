import os
import time
import json
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
import mlx.nn as nn
import streamlit as st

@dataclass
class InferenceParams:
    """Dynamic inference parameters that can be hot-swapped"""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 2048
    
    # Advanced parameters
    guidance_scale: float = 1.0
    dynamic_temperature: bool = False
    temp_decay: float = 0.95
    context_window_shift: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        params = {k: v for k, v in asdict(self).items() if v is not None}
        
        # Convert parameter names to match make_sampler expectations
        if 'temperature' in params:
            params['temp'] = params.pop('temperature')
        
        # Remove parameters that make_sampler doesn't accept
        excluded_params = {
            'max_tokens', 'guidance_scale', 'dynamic_temperature', 
            'temp_decay', 'context_window_shift', 'frequency_penalty', 
            'presence_penalty', 'repetition_penalty'
        }
        for param in excluded_params:
            params.pop(param, None)
        
        return params

@dataclass
class FeedbackSignal:
    """Feedback signal for model adjustment"""
    response_quality: float  # 0-1 score
    coherence: float
    relevance: float
    creativity: float
    factuality: float
    user_satisfaction: float
    timestamp: float
    
    def composite_score(self) -> float:
        weights = {
            'quality': 0.3,
            'coherence': 0.2,
            'relevance': 0.2,
            'creativity': 0.1,
            'factuality': 0.1,
            'satisfaction': 0.1
        }
        return (
            weights['quality'] * self.response_quality +
            weights['coherence'] * self.coherence +
            weights['relevance'] * self.relevance +
            weights['creativity'] * self.creativity +
            weights['factuality'] * self.factuality +
            weights['satisfaction'] * self.user_satisfaction
        )

class DynamicLoRAAdapter:
    """Lightweight LoRA adapter that can be hot-swapped"""
    
    def __init__(self, rank: int = 8, alpha: float = 16):
        self.rank = rank
        self.alpha = alpha
        self.adapters: Dict[str, Tuple[mx.array, mx.array]] = {}
        self.active_adapters: List[str] = []
        self.adapter_weights: Dict[str, float] = {}
    
    def create_adapter(self, name: str, layer_shapes: Dict[str, Tuple[int, int]]):
        """Create a new LoRA adapter without reloading the base model"""
        adapters = {}
        for layer_name, (in_dim, out_dim) in layer_shapes.items():
            # Initialize LoRA matrices A and B
            lora_a = mx.random.normal((in_dim, self.rank)) * 0.01
            lora_b = mx.random.normal((self.rank, out_dim)) * 0.01
            adapters[layer_name] = (lora_a, lora_b)
        
        self.adapters[name] = adapters
        self.adapter_weights[name] = 1.0
        return name
    
    def blend_adapters(self, adapter_names: List[str], weights: List[float]):
        """Hot-blend multiple adapters without model reload"""
        self.active_adapters = adapter_names
        for name, weight in zip(adapter_names, weights):
            self.adapter_weights[name] = weight
    
    def apply_adapter_delta(self, x: mx.array, layer_name: str) -> mx.array:
        """Apply adapter transformation to layer input"""
        if not self.active_adapters:
            return x
        
        delta = mx.zeros_like(x)
        for adapter_name in self.active_adapters:
            if adapter_name in self.adapters and layer_name in self.adapters[adapter_name]:
                lora_a, lora_b = self.adapters[adapter_name][layer_name]
                weight = self.adapter_weights[adapter_name]
                # LoRA: x + (x @ A @ B) * (alpha/rank) * weight
                adapter_output = x @ lora_a @ lora_b
                delta = delta + adapter_output * (self.alpha / self.rank) * weight
        
        return x + delta

class HotModelInference:
    """Hot Model Inference system with real-time feedback loops"""
    
    def __init__(self, model_path: str, session_id: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.base_params = InferenceParams()
        self.current_params = InferenceParams()
        self.lora_adapter = DynamicLoRAAdapter()
        
        # Session management
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = os.path.join(os.path.dirname(model_path), "hmi_sessions")
        self.session_file = os.path.join(self.session_dir, f"{self.session_id}.pkl")
        
        # Feedback system
        self.feedback_history: List[FeedbackSignal] = []
        self.adaptation_threshold = 0.1
        self.learning_rate = 0.01
        
        # Performance tracking
        self.generation_cache = {}
        self.param_performance: Dict[str, List[float]] = {}
        
        # Auto-save settings
        self.auto_save_enabled = True
        self.save_interval = 10  # Save every 10 feedback entries
        self.last_save_time = time.time()
        
        # Create session directory
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Load model and try to restore session
        self._load_model()
        self._try_restore_session()
    
    def _load_model(self):
        """Load the base model (only done once)"""
        print(f"Loading base model: {self.model_path}")
        self.model, self.tokenizer = load(self.model_path, lazy=True)
        print("Base model loaded successfully")
    
    def _try_restore_session(self):
        """Try to restore a previous session"""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'rb') as f:
                    session_data = pickle.load(f)
                
                # Restore parameters
                if 'current_params' in session_data:
                    self.current_params = InferenceParams(**session_data['current_params'])
                
                # Restore feedback history
                if 'feedback_history' in session_data:
                    self.feedback_history = [
                        FeedbackSignal(**fb) for fb in session_data['feedback_history']
                    ]
                
                # Restore LoRA adapters
                if 'lora_adapters' in session_data:
                    self.lora_adapter.adapters = session_data['lora_adapters']
                    self.lora_adapter.active_adapters = session_data.get('active_adapters', [])
                    self.lora_adapter.adapter_weights = session_data.get('adapter_weights', {})
                
                # Restore performance data
                if 'param_performance' in session_data:
                    self.param_performance = session_data['param_performance']
                
                print(f"Session restored: {self.session_id} ({len(self.feedback_history)} feedback entries)")
                
            except Exception as e:
                print(f"Failed to restore session {self.session_id}: {e}")
                # Continue with fresh session
        else:
            print(f"Starting new session: {self.session_id}")
    
    def _save_session(self, force: bool = False):
        """Save current session state"""
        if not self.auto_save_enabled and not force:
            return
        
        # Check if it's time to auto-save
        time_since_save = time.time() - self.last_save_time
        feedback_since_save = len(self.feedback_history) % self.save_interval == 0
        
        if not force and not (feedback_since_save and time_since_save > 30):
            return
        
        try:
            session_data = {
                'session_id': self.session_id,
                'model_path': self.model_path,
                'timestamp': datetime.now().isoformat(),
                'current_params': asdict(self.current_params),
                'feedback_history': [asdict(fb) for fb in self.feedback_history],
                'lora_adapters': self.lora_adapter.adapters,
                'active_adapters': self.lora_adapter.active_adapters,
                'adapter_weights': self.lora_adapter.adapter_weights,
                'param_performance': self.param_performance,
                'generation_cache_size': len(self.generation_cache)
            }
            
            with open(self.session_file, 'wb') as f:
                pickle.dump(session_data, f)
            
            self.last_save_time = time.time()
            print(f"Session saved: {self.session_id}")
            
        except Exception as e:
            print(f"Failed to save session: {e}")
    
    def list_available_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions for this model"""
        sessions = []
        
        if not os.path.exists(self.session_dir):
            return sessions
        
        for filename in os.listdir(self.session_dir):
            if filename.endswith('.pkl'):
                session_path = os.path.join(self.session_dir, filename)
                try:
                    with open(session_path, 'rb') as f:
                        session_data = pickle.load(f)
                    
                    sessions.append({
                        'session_id': session_data.get('session_id', 'unknown'),
                        'timestamp': session_data.get('timestamp', 'unknown'),
                        'feedback_count': len(session_data.get('feedback_history', [])),
                        'adapter_count': len(session_data.get('lora_adapters', {})),
                        'file_path': session_path
                    })
                except Exception as e:
                    print(f"Failed to read session {filename}: {e}")
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        return sessions
    
    def load_session(self, session_id: str) -> bool:
        """Load a specific session"""
        session_file = os.path.join(self.session_dir, f"{session_id}.pkl")
        
        if not os.path.exists(session_file):
            print(f"Session {session_id} not found")
            return False
        
        # Save current session first
        self._save_session(force=True)
        
        # Load the specified session
        self.session_id = session_id
        self.session_file = session_file
        self._try_restore_session()
        
        return True
    
    def hot_update_params(self, **kwargs) -> InferenceParams:
        """Hot-update inference parameters without model reload"""
        old_params = self.current_params
        
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self.current_params, key):
                setattr(self.current_params, key, value)
        
        # Log parameter change
        changed_params = {
            k: v for k, v in kwargs.items() 
            if hasattr(old_params, k) and getattr(old_params, k) != v
        }
        
        if changed_params:
            print(f"Hot-updated parameters: {changed_params}")
        
        return self.current_params
    
    def _extract_model_layer_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Extract layer shapes from the loaded model dynamically"""
        layer_shapes = {}
        
        if self.model is None:
            # Fallback to common shapes if model not loaded
            return {
                "attention.query": (4096, 4096),
                "attention.key": (4096, 4096), 
                "attention.value": (4096, 4096),
                "mlp.gate": (4096, 11008),
                "mlp.up": (4096, 11008),
                "mlp.down": (11008, 4096)
            }
        
        try:
            # Inspect MLX model layers to get actual dimensions
            print(f"Debug: Model type: {type(self.model)}")
            print(f"Debug: Model attributes: {list(self.model.__dict__.keys()) if hasattr(self.model, '__dict__') else 'No __dict__'}")
            
            # Try different MLX model introspection methods
            if hasattr(self.model, 'layers') and self.model.layers:
                print(f"Debug: Found model.layers with {len(self.model.layers)} layers")
                # Get first layer to inspect structure
                first_layer = self.model.layers[0]
                print(f"Debug: First layer type: {type(first_layer)}")
                print(f"Debug: First layer attributes: {list(first_layer.__dict__.keys()) if hasattr(first_layer, '__dict__') else 'No __dict__'}")
                
                # Try to find attention and MLP components
                if hasattr(first_layer, 'attention') or hasattr(first_layer, 'self_attn'):
                    attn = getattr(first_layer, 'attention', getattr(first_layer, 'self_attn', None))
                    if attn:
                        print(f"Debug: Attention attributes: {list(attn.__dict__.keys()) if hasattr(attn, '__dict__') else 'No __dict__'}")
                        
                        # Check for different attention weight naming conventions
                        for attr_name in ['q_proj', 'query', 'wq', 'query_proj']:
                            if hasattr(attn, attr_name):
                                q_layer = getattr(attn, attr_name)
                                if hasattr(q_layer, 'weight'):
                                    layer_shapes["attention.query"] = tuple(q_layer.weight.shape)
                                    print(f"Debug: Found query weights shape: {layer_shapes['attention.query']}")
                                    break
                        
                        for attr_name in ['k_proj', 'key', 'wk', 'key_proj']:
                            if hasattr(attn, attr_name):
                                k_layer = getattr(attn, attr_name)
                                if hasattr(k_layer, 'weight'):
                                    layer_shapes["attention.key"] = tuple(k_layer.weight.shape)
                                    print(f"Debug: Found key weights shape: {layer_shapes['attention.key']}")
                                    break
                                    
                        for attr_name in ['v_proj', 'value', 'wv', 'value_proj']:
                            if hasattr(attn, attr_name):
                                v_layer = getattr(attn, attr_name)
                                if hasattr(v_layer, 'weight'):
                                    layer_shapes["attention.value"] = tuple(v_layer.weight.shape)
                                    print(f"Debug: Found value weights shape: {layer_shapes['attention.value']}")
                                    break
                
                # Try to find MLP components
                if hasattr(first_layer, 'mlp') or hasattr(first_layer, 'feed_forward'):
                    mlp = getattr(first_layer, 'mlp', getattr(first_layer, 'feed_forward', None))
                    if mlp:
                        print(f"Debug: MLP attributes: {list(mlp.__dict__.keys()) if hasattr(mlp, '__dict__') else 'No __dict__'}")
                        
                        for attr_name in ['gate_proj', 'gate', 'w1', 'up_gate']:
                            if hasattr(mlp, attr_name):
                                gate_layer = getattr(mlp, attr_name)
                                if hasattr(gate_layer, 'weight'):
                                    layer_shapes["mlp.gate"] = tuple(gate_layer.weight.shape)
                                    print(f"Debug: Found gate weights shape: {layer_shapes['mlp.gate']}")
                                    break
                                    
                        for attr_name in ['up_proj', 'up', 'w3', 'up_linear']:
                            if hasattr(mlp, attr_name):
                                up_layer = getattr(mlp, attr_name)
                                if hasattr(up_layer, 'weight'):
                                    layer_shapes["mlp.up"] = tuple(up_layer.weight.shape)
                                    print(f"Debug: Found up weights shape: {layer_shapes['mlp.up']}")
                                    break
                                    
                        for attr_name in ['down_proj', 'down', 'w2', 'down_linear']:
                            if hasattr(mlp, attr_name):
                                down_layer = getattr(mlp, attr_name)
                                if hasattr(down_layer, 'weight'):
                                    layer_shapes["mlp.down"] = tuple(down_layer.weight.shape)
                                    print(f"Debug: Found down weights shape: {layer_shapes['mlp.down']}")
                                    break
            
            print(f"Debug: Final extracted layer shapes: {layer_shapes}")
            
        except Exception as e:
            print(f"Could not extract model shapes, using defaults: {e}")
            # Fallback to default shapes
            layer_shapes = {
                "attention.query": (4096, 4096),
                "attention.key": (4096, 4096), 
                "attention.value": (4096, 4096),
                "mlp.gate": (4096, 11008),
                "mlp.up": (4096, 11008),
                "mlp.down": (11008, 4096)
            }
        
        return layer_shapes
    
    def create_feedback_adapter(self, feedback_signals: List[FeedbackSignal]) -> str:
        """Create a new LoRA adapter based on feedback signals"""
        adapter_name = f"feedback_adapter_{int(time.time())}"
        
        # Analyze feedback patterns
        avg_score = np.mean([f.composite_score() for f in feedback_signals])
        
        if avg_score < 0.7:  # Poor performance, create corrective adapter
            # Get dynamic layer shapes from the actual model
            layer_shapes = self._extract_model_layer_shapes()
            
            self.lora_adapter.create_adapter(adapter_name, layer_shapes)
            print(f"Created feedback adapter: {adapter_name} (score: {avg_score:.3f})")
            
        return adapter_name
    
    def adaptive_generate(self, prompt: str, feedback_callback=None):
        """Generate with adaptive parameters based on real-time feedback - STREAMING"""
        
        # Create sampler with current parameters
        sampler_params = self.current_params.to_dict()
        sampler = make_sampler(**sampler_params)
        
        # Track generation metrics
        start_time = time.time()
        tokens_generated = 0
        response_text = ""
        
        # Stream generation with adaptive adjustments
        for token in stream_generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=self.current_params.max_tokens,
            sampler=sampler
        ):
            token_text = token.text if hasattr(token, 'text') else token
            response_text += token_text
            tokens_generated += 1
            
            # Yield the token for streaming
            yield token
            
            # Dynamic temperature adjustment
            if self.current_params.dynamic_temperature and tokens_generated % 10 == 0:
                new_temp = max(0.1, self.current_params.temperature * self.current_params.temp_decay)
                self.hot_update_params(temperature=new_temp)
                sampler = make_sampler(**self.current_params.to_dict())
            
            # Real-time feedback callback
            if feedback_callback and tokens_generated % 20 == 0:
                feedback = feedback_callback(response_text)
                if feedback and feedback.composite_score() < 0.5:
                    # Emergency parameter adjustment
                    self.hot_update_params(
                        temperature=min(1.5, self.current_params.temperature * 1.2),
                        top_p=max(0.8, self.current_params.top_p * 0.95)
                    )
                    sampler = make_sampler(**self.current_params.to_dict())
        
        # Record performance after streaming is complete
        generation_time = time.time() - start_time
        tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Cache for performance analysis
        cache_key = f"{hash(prompt)}_{hash(str(sampler_params))}"
        self.generation_cache[cache_key] = {
            'response': response_text,
            'params': sampler_params,
            'performance': {
                'tokens_generated': tokens_generated,
                'generation_time': generation_time,
                'tokens_per_sec': tokens_per_sec
            }
        }
    
    def learn_from_feedback(self, response: str, feedback: FeedbackSignal):
        """Learn and adapt from user feedback"""
        self.feedback_history.append(feedback)
        
        # Adaptive parameter adjustment based on feedback
        score = feedback.composite_score()
        
        if score < 0.6:  # Poor response
            # Increase creativity and reduce repetition
            self.hot_update_params(
                temperature=min(1.2, self.current_params.temperature * 1.1),
                repetition_penalty=min(1.5, self.current_params.repetition_penalty * 1.1),
                top_p=max(0.85, self.current_params.top_p * 0.95)
            )
        elif score > 0.8:  # Great response
            # Reinforce current parameters
            self.hot_update_params(
                temperature=max(0.3, self.current_params.temperature * 0.98),
                repetition_penalty=max(1.0, self.current_params.repetition_penalty * 0.98)
            )
        
        # Create new adapter if we have enough feedback
        if len(self.feedback_history) >= 1:
            recent_feedback = self.feedback_history[-1:]
            adapter_name = self.create_feedback_adapter(recent_feedback)
            
            # Activate the new adapter
            if adapter_name:
                self.lora_adapter.blend_adapters([adapter_name], [0.1])
        
        # Auto-save session after feedback
        self._save_session()
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights on parameter performance"""
        if not self.feedback_history:
            return {"message": "No feedback data available"}
        
        recent_scores = [f.composite_score() for f in self.feedback_history[-20:]]
        
        # Calculate trend safely
        score_trend = 0.0  # Default to no trend
        if len(recent_scores) >= 2:
            try:
                score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            except np.linalg.LinAlgError:
                # SVD did not converge, use simple difference as fallback
                score_trend = recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0.0
        
        return {
            "avg_score": np.mean(recent_scores) if recent_scores else 0.0,
            "score_trend": score_trend,
            "best_params": self._find_best_params(),
            "feedback_count": len(self.feedback_history),
            "active_adapters": self.lora_adapter.active_adapters
        }
    
    def _find_best_params(self) -> Dict[str, Any]:
        """Find the best performing parameter combination"""
        if not self.generation_cache:
            return {}
        
        best_score = 0
        best_params = {}
        
        for cache_entry in self.generation_cache.values():
            # Simple heuristic: tokens_per_sec as performance metric
            score = cache_entry['performance']['tokens_per_sec']
            if score > best_score:
                best_score = score
                best_params = cache_entry['params']
        
        return best_params
    
    def reset_to_baseline(self):
        """Reset to baseline parameters"""
        self.current_params = InferenceParams()
        self.lora_adapter.active_adapters = []
        print("Reset to baseline parameters")
    
    def save_adaptation_state(self, filepath: str):
        """Save current adaptation state"""
        state = {
            'current_params': asdict(self.current_params),
            'feedback_history': [asdict(f) for f in self.feedback_history],
            'performance_insights': self.get_performance_insights()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Adaptation state saved to {filepath}")
    
    def load_adaptation_state(self, filepath: str):
        """Load adaptation state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore parameters
        self.current_params = InferenceParams(**state['current_params'])
        
        # Restore feedback history
        self.feedback_history = [
            FeedbackSignal(**f) for f in state['feedback_history']
        ]
        
        print(f"Adaptation state loaded from {filepath}")

# Example usage and integration functions

def create_feedback_ui():
    """Create Streamlit UI for real-time feedback"""
    st.subheader("🔄 Real-time Model Feedback")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quality = st.slider("Response Quality", 0.0, 1.0, 0.7, 0.1)
        coherence = st.slider("Coherence", 0.0, 1.0, 0.7, 0.1)
    
    with col2:
        relevance = st.slider("Relevance", 0.0, 1.0, 0.7, 0.1)
        creativity = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)
    
    with col3:
        factuality = st.slider("Factuality", 0.0, 1.0, 0.7, 0.1)
        satisfaction = st.slider("Satisfaction", 0.0, 1.0, 0.7, 0.1)
    
    if st.button("Submit Feedback"):
        feedback = FeedbackSignal(
            response_quality=quality,
            coherence=coherence,
            relevance=relevance,
            creativity=creativity,
            factuality=factuality,
            user_satisfaction=satisfaction,
            timestamp=time.time()
        )
        return feedback
    
    return None

def create_hmi_control_panel(hmi: HotModelInference):
    """Create control panel for Hot Model Inference"""
    
    # Session management
    with st.expander("📂 Session Management", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Current Session:** `{hmi.session_id}`")
            
            # Auto-save toggle
            auto_save = st.checkbox(
                "Auto-save sessions", 
                value=hmi.auto_save_enabled,
                help="Automatically save LoRA adapters and learned parameters"
            )
            hmi.auto_save_enabled = auto_save
            
            if st.button("💾 Force Save Session"):
                hmi._save_session(force=True)
                st.success("Session saved!")
        
        with col2:
            # List available sessions
            sessions = hmi.list_available_sessions()
            if sessions:
                st.markdown("**Available Sessions:**")
                for session in sessions[:5]:  # Show last 5 sessions
                    session_info = f"{session['session_id']} - {session['feedback_count']} feedback, {session['adapter_count']} adapters"
                    if st.button(f"📁 {session_info}", key=f"load_{session['session_id']}"):
                        if hmi.load_session(session['session_id']):
                            st.success(f"Loaded session: {session['session_id']}")
                            st.rerun()
            else:
                st.info("No previous sessions found")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dynamic Parameters**")
        new_temp = st.slider("Temperature", 0.1, 2.0, hmi.current_params.temperature, 0.1)
        new_top_p = st.slider("Top-p", 0.1, 1.0, hmi.current_params.top_p, 0.05)
        new_rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, hmi.current_params.repetition_penalty, 0.1)
        
        dynamic_temp = st.checkbox("Dynamic Temperature", hmi.current_params.dynamic_temperature)
        
        if st.button("Hot Update Parameters"):
            hmi.hot_update_params(
                temperature=new_temp,
                top_p=new_top_p,
                repetition_penalty=new_rep_penalty,
                dynamic_temperature=dynamic_temp
            )
            st.success("Parameters updated in real-time!")
    
    with col2:
        st.markdown("**Performance Insights**")
        insights = hmi.get_performance_insights()
        
        if insights.get('avg_score'):
            st.metric("Average Score", f"{insights['avg_score']:.3f}")
            st.metric("Feedback Count", insights['feedback_count'])
            st.metric("Active Adapters", len(insights.get('active_adapters', [])))
            
            if insights['score_trend'] > 0:
                st.success(f"📈 Improving (+{insights['score_trend']:.3f})")
            else:
                st.warning(f"📉 Declining ({insights['score_trend']:.3f})")
        else:
            # Show basic info even when no feedback is available
            st.metric("Feedback Count", insights.get('feedback_count', 0))
            st.metric("Active Adapters", len(insights.get('active_adapters', [])))
            if insights.get('message'):
                st.info(insights['message'])
        
        if st.button("Reset to Baseline"):
            hmi.reset_to_baseline()
            st.success("Reset to baseline parameters")
    
    return hmi
