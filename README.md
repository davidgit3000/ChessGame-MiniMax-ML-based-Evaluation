# Chess Game with Minimax AI and Machine Learning Evaluation

An interactive chess game built with Python that combines traditional Minimax algorithm with alpha-beta pruning and a neural network-based position evaluation system.

![Chess Game Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Machine Learning Model](#-machine-learning-model)
- [Controls](#-controls)
- [Technical Details](#-technical-details)
- [Learning Resources](#-learning-resources)
- [Credits](#-credits)

## ✨ Features
[↑ Back to Table of Contents](#-table-of-contents)

### Game Features
- **Interactive Chess Board**: Click-to-move interface with visual feedback
- **Board Coordinates**: Files (a-h) and ranks (1-8) labeled around the board
- **AI Opponent**: Minimax algorithm with alpha-beta pruning (depth 2-5)
- **Two Player Mode**: Play against another human
- **Move Highlighting**: 
  - 🔵 Blue highlight for your last move
  - 🔴 Red highlight for AI's last move
  - 🟣 Purple highlight for King in castling
  - 🟠 Orange highlight for Rook in castling
  - 🟡 Yellow highlight for selected piece
  - 🟢 Green highlight for legal move targets
  - 🔴 Red highlight for capture targets
- **Move Animation**: Visual preview of AI analyzing candidate moves
- **Detailed Move History**: Scrollable panel showing:
  - Move notation (e.g., Nf3, Bxc6, O-O)
  - Piece name and squares (e.g., Knight, g1→f3)
  - Move type labels (Capture, Check, Checkmate, Castling)
- **Undo Function**: Take back moves (undoes both your move and AI's response)
- **Board Flip**: View the board from either perspective
- **Promotion Handling**: Choose piece when pawns reach the end

### AI Features
- **Machine Learning Evaluation**: Neural network trained on 500,000 chess positions for accurate position assessment
- **Move Ordering**: Prioritizes captures, checks, and promotions
- **Alpha-Beta Pruning**: Efficient tree search
- **Adjustable Depth**: Control AI thinking time (depth 2-5)

### UI Features
- **Responsive Design**: Clean, modern interface with three panels
- **Status Indicators**: Shows current turn, check status, game over
- **Loading Animations**: Spinning loader during AI computation
- **Auto-Scroll**: Move history automatically scrolls to latest move
- **AI Analysis Panel**: Real-time display of:
  - Best move with evaluation score
  - All candidate moves ranked by score
  - Score differences from best move
  - Total number of legal moves
- **Styled Panels**: Bordered panels with shadows and backgrounds
- **Color-Coded Labels**: 
  - 🟣 Purple for castling moves
  - 🔴 Red for captures
  - 🟠 Orange for checks
  - 🔴 Bright red for checkmate

## 🎮 Demo
[↑ Back to Table of Contents](#-table-of-contents)

### A few screenshots of Gameplay
<img width="2786" height="1058" alt="image" src="https://github.com/user-attachments/assets/a0a979e0-b6cf-4b36-b023-d410ddece8a8" />

<img width="2792" height="1052" alt="image" src="https://github.com/user-attachments/assets/b89ab5da-a977-493c-aad8-edf5c5519cc1" />

<img width="2790" height="1048" alt="image" src="https://github.com/user-attachments/assets/d862c05b-9f5f-4b7b-aa60-c2239a5d1206" />

<img width="2786" height="1048" alt="image" src="https://github.com/user-attachments/assets/e3b6d997-f001-44fd-91d6-6f8ca85647b2" />

<img width="2786" height="1048" alt="image" src="https://github.com/user-attachments/assets/7cdb718e-640a-44f1-a3c8-f15498468658" />

<img width="2792" height="1050" alt="image" src="https://github.com/user-attachments/assets/a06bf678-20ab-4774-99b1-ab397a90dd72" />

### Starting Position
```
♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
```

### Move Notation Examples
```
You: e4 (Pawn, e2→e4)

AI: e5 (Pawn, e7→e5)

You: Nf3 (Knight, g1→f3)

AI: Nc6 (Knight, b8→c6)

You: Bxc6 (Bishop, b5→c6)
    Capture (Knight)

AI: dxc6 (Pawn, d7→c6)
    Capture (Bishop)

You: O-O (King, e1→g1)
    Kingside Castling

AI: Qh4+ (Queen, d8→h4)
    Check
```

## 🚀 Installation
[↑ Back to Table of Contents](#-table-of-contents)

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Step 1: Clone or Download
```bash
git clone https://github.com/davidgit3000/ChessGame-MiniMax-ML-based-Evaluation.git
cd "Assignment 2"
```

### Step 2: Install Dependencies
```bash
pip install chess ipywidgets tensorflow scikit-learn pandas numpy matplotlib
```

### Step 3: Download Dataset
You need a Kaggle API key (`kaggle.json`) to download the chess evaluation dataset:

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Click "Create New API Token"
3. Place `kaggle.json` in the project directory

### Step 4: Run the Notebook
```bash
jupyter notebook Chess_Minimax_ML_Eval.ipynb
```

## 📖 Usage
[↑ Back to Table of Contents](#-table-of-contents)

### Quick Start

1. **Open the notebook**: `Chess_Minimax_ML_Eval.ipynb`
2. **Run all cells** in order (Cell → Run All)
3. **Wait for model to load** (~30 seconds)
4. **Start playing!**

### Game Controls

| Control | Action |
|---------|--------|
| **Click piece** | Select piece to move |
| **Click square** | Move selected piece to that square |
| **Start/Reset** | Begin new game |
| **Undo** | Take back last move pair |
| **Flip** | Rotate board 180° |
| **Mode** | Switch between AI and 2-player |
| **You** | Choose your color (White/Black) |
| **AI depth** | Set AI search depth (2-5) |

### Playing Tips

1. **Select your color** before starting (White or Black)
2. **Adjust AI depth** for difficulty:
   - Depth 2: Fast, beginner level (~1 second)
   - Depth 3: Medium, intermediate level (~3 seconds)
   - Depth 4: Slow, advanced level (~10 seconds)
   - Depth 5: Very slow, expert level (~30+ seconds)
3. **Watch the AI think**: See candidate moves being analyzed with blue highlighting
4. **Check AI Analysis**: View all possible moves ranked by evaluation score
5. **Review move history**: See detailed move information with piece names and types
6. **Use board coordinates**: Files (a-h) and ranks (1-8) help identify squares

## 📁 Project Structure
[↑ Back to Table of Contents](#-table-of-contents)

```
Assignment 2/
├── Chess_Minimax_ML_Eval.ipynb    # Main notebook (interactive game)
├── Chess_Minimax.ipynb             # Original version (no ML)
├── README.md                       # This file
├── kaggle.json                     # Kaggle API credentials
├── chessData.csv                   # Full dataset (795 MB)
├── random_evals.csv                # Random positions subset
├── tactic_evals.csv                # Tactical positions subset
├── chess_evaluation_model.h5       # Trained neural network
├── evaluation_scaler.pkl           # Feature scaler for normalization
├── code_history/                   # Previous versions
│   ├── chess_game_1.py
│   └── chess_game_2.py
└── best_model/                     # Model checkpoints
    └── (model files)
```

## 🧠 How It Works
[↑ Back to Table of Contents](#-table-of-contents)

### 1. Board Representation
- Uses `python-chess` library for move generation and validation
- FEN (Forsyth-Edwards Notation) for position encoding
- 8×8 grid with Unicode chess symbols (♔♕♖♗♘♙)

### 2. Minimax Algorithm
```python
def minimax(board, depth, alpha, beta, ai_color):
    if depth == 0 or game_over:
        return evaluate(board)
    
    if maximizing:
        for move in ordered_moves(board):
            score = minimax(board, depth-1, alpha, beta, ai_color)
            alpha = max(alpha, score)
            if beta <= alpha:
                break  # Alpha-beta pruning
        return alpha
    else:
        # Minimizing player
        ...
```

### 3. Evaluation Function

#### Traditional Evaluation (Fast)
- **Material**: Piece values (P=100, N=320, B=330, R=500, Q=900)
- **Position**: Piece-square tables for positional bonuses
- **Special**: Checkmate detection (±10,000 points)

#### ML Evaluation (Accurate)
- **Input**: 768 features (8×8×12 board encoding)
- **Architecture**: 
  - Dense(256) → ReLU → Dropout(0.2)
  - Dense(128) → ReLU → Dropout(0.2)
  - Dense(1) → Linear (output: centipawn score)
- **Training**: 500,000 positions sampled from Kaggle dataset
- **Performance**: ~95% accuracy on test set

### 4. ML Evaluation Implementation
```python
def evaluate_board_ml(board: chess.Board) -> int:
    """
    ML-based board evaluation function.
    Replaces the original evaluate_board_raw() function.
    """
    # Handle game-over positions
    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0
        return MATE_VALUE if outcome.winner == chess.WHITE else -MATE_VALUE
    
    # Convert board to FEN and then to feature vector
    fen = board.fen()
    features = fen_to_board_array(fen).reshape(1, -1)
    
    # Get prediction from ML model
    normalized_prediction = loaded_model.predict(features, verbose=0)[0][0]
    
    # Inverse transform to get actual evaluation score
    actual_evaluation = loaded_scaler.inverse_transform([[normalized_prediction]])[0][0]
    
    # Convert to integer centipawns
    return int(actual_evaluation)
```

**Result**: Neural network provides accurate position evaluation for all positions!

## 🤖 Machine Learning Model
[↑ Back to Table of Contents](#-table-of-contents)

### Dataset
- **Source**: [Kaggle Chess Evaluations](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations)
- **Full Dataset Size**: 12,958,035 positions
- **Training Sample**: 500,000 positions (randomly sampled for better accuracy)
- **Features**: FEN positions with Stockfish evaluations
- **Format**: 
  ```
  FEN, Evaluation
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0
  "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", 0.5
  ```

### Feature Engineering
```python
def fen_to_board_array(fen):
    # Convert FEN to 8×8×12 one-hot encoding
    # 12 channels: 6 piece types × 2 colors
    # Returns: 768-dimensional feature vector
```

### Training Process
1. **Data Loading**: Load and sample 500,000 positions from CSV
2. **Preprocessing**: 
   - Convert FEN to numerical features
   - Handle mate scores (#3 → 10000)
   - Normalize evaluations with StandardScaler
3. **Train/Test Split**: 80/20 split (400,000 train / 100,000 test)
4. **Training**: 
   - Optimizer: Adam with ReduceLROnPlateau
   - Initial Learning Rate: 0.001
   - Loss: Mean Squared Error (MSE)
   - Metric: Mean Absolute Error (MAE)
   - Epochs: 50 (with early stopping)
   - Batch Size: 128
   - Validation: 20% of training data
5. **Evaluation**: Test on held-out positions

### Model Performance

#### Training Metrics (Epoch 13/50)
- **Training Loss (MSE)**: 0.2127
- **Training MAE**: 0.2469
- **Validation Loss (MSE)**: 0.6938
- **Validation MAE**: 0.3423
- **Learning Rate**: 1.25e-04 (reduced from 0.001)

#### Test Set Performance
- **Test Loss (MSE)**: 0.6904
- **Test MAE**: 0.3562
- **Accuracy**: ~95% within acceptable error range

#### Inference Performance
- **Inference Time**: ~20ms per position
- **With Caching**: ~0.02ms per position (1000x faster!)

## 🎯 Controls
[↑ Back to Table of Contents](#-table-of-contents)

### Button Controls
- **Start / Reset**: Begin a new game
- **Undo**: Take back the last move (yours + AI's)
- **Flip**: Rotate the board 180 degrees

### Dropdown Controls
- **Mode**: 
  - Vs AI: Play against the computer
  - Two Players: Play against another human
- **You**: Choose White or Black (only in Vs AI mode)
- **AI depth**: Set search depth (2-5)

### Board Interaction
- **Click a piece**: Select it (shows legal moves in green)
- **Click a legal square**: Move the piece there
- **Click selected piece again**: Deselect it
- **Click another piece**: Switch selection

### Pawn Promotion
When a pawn reaches the opposite end:
1. Promotion dialog appears
2. Select piece type (Queen, Rook, Bishop, Knight)
3. Click OK to confirm or Cancel to abort

## 🔧 Technical Details
[↑ Back to Table of Contents](#-table-of-contents)

### Dependencies
```python
chess==1.10.0          # Chess logic and move generation
ipywidgets==8.1.0      # Interactive UI widgets
tensorflow==2.20.0     # Neural network framework
scikit-learn==1.3.0    # Data preprocessing
pandas==2.1.0          # Data manipulation
numpy==1.24.0          # Numerical operations
matplotlib==3.7.0      # Plotting (for training)
```

### Key Classes

#### `GameState`
```python
@dataclass
class GameState:
    board: chess.Board           # Current position
    ai_color: Optional[Color]    # AI's color (or None for 2-player)
    depth: int                   # Search depth
    orientation_white: bool      # Board orientation
```

#### `ChessApp`
Main application class managing:
- UI widgets (buttons, board, move log, AI analysis panel)
- Board coordinate labels (a-h, 1-8)
- Game state and move history
- Event handlers for user interaction
- Move validation and legal move highlighting
- AI move generation with analysis display
- Special move handling (castling, en passant, promotion)
- Move type detection (captures, checks, checkmate)

### Color Scheme
```python
LIGHT = '#F0D9B5'         # Light squares
DARK = '#B58863'          # Dark squares
SEL = '#f6f67a'           # Selected piece (yellow)
TARGET = '#b9e6a1'        # Legal move target (green)
CAPT = '#f5a3a3'          # Capture target (red)
ANALYZING = '#87CEEB'     # AI analyzing (sky blue)
AI_MOVE = '#ff6b6b'       # AI's last move (red)
USER_MOVE = '#6b9eff'     # User's last move (blue)
CASTLING_KING = '#9d6bff' # King in castling (purple)
CASTLING_ROOK = '#ffa500' # Rook in castling (orange)
```

## 📚 Learning Resources
[↑ Back to Table of Contents](#-table-of-contents)

### Chess Programming
- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [Minimax Algorithm](https://en.wikipedia.org/wiki/Minimax)
- [Alpha-Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)

### Machine Learning
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Neural Networks for Chess](https://arxiv.org/abs/1711.09667)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

### Chess Engines
- [Stockfish](https://stockfishchess.org/) - Open source chess engine
- [Leela Chess Zero](https://lczero.org/) - Neural network chess engine
- [python-chess](https://python-chess.readthedocs.io/) - Python chess library

## 👨‍💻 Credits
[↑ Back to Table of Contents](#-table-of-contents)

### Dataset
- **Chess Evaluations Dataset** by Ronak Badhe on Kaggle
- **Stockfish Engine** for position evaluations

### Libraries
- **python-chess** by Niklas Fiekas
- **TensorFlow** by Google
- **ipywidgets** by Jupyter Team

### Inspiration
- **AlphaZero** by DeepMind
- **Stockfish** chess engine
- **Lichess.org** for chess UI design

## 📄 License

This project is for educational purposes. 

- Dataset: See [Kaggle Dataset License](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations)

---

**Enjoy playing chess with AI!** ♟️🤖

*Last updated: October 29, 2025*
