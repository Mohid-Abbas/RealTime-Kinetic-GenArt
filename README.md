# KineticGhost.ai: Real-Time Generative Motion Art
*"Turning Human Kinematics into a Symphony of Light."*

KineticGhost is a high-performance Computer Vision installation that bridges the gap between AI-driven pose estimation and generative digital art. By leveraging MediaPipe’s blazingly fast landmark detection and a custom 2D physics engine, this project transforms a live webcam feed into a dynamic, ethereal particle system.

## ✨ The "Fascinating" Vision
The goal was simple: Make the invisible, visible. The system doesn't just "detect" a body; it reimagines it. Every gesture creates a digital aura, and every movement leaves a neon trail in its wake. It is designed to be a high-visual-impact showcase of what happens when Artificial Intelligence meets Creative Coding.

## 🛠️ The Tech Stack (Under the Hood)
- **Core Logic**: Python & OpenCV
- **AI Engine**: MediaPipe (Holistic Pose Estimation)
- **Physics Engine**: Custom-built vectorized NumPy particle dynamics (Velocity, Friction, Gravity)
- **Temporal Analysis**: Motion History Imaging (MHI) via coordinate queueing for real-time light trails.
- **Rendering**: Optimized NumPy index mapping and Additive blending for a high-FPS "glow" effect.

## 🚀 Key Technical Breakthroughs
- **Zero-Latency Pipeline**: Implemented a highly optimized NumPy-based renderer that updates and draws 20,000+ particles at 30+ FPS without traditional loop bottlenecks.
- **Dynamic Motion Trails**: Implemented a fast decaying temporal buffer that creates glowing arcs based on the velocity of user gestures.
- **Jitter-Free Tracking**: Utilizes a Moving Average smoothing filter on MediaPipe landmarks to prevent AI "flicker" and create a silky smooth visual output.
- **Reactive Aesthetic**: The particle field responds to the intensity of motion. Explosive hand movements spawn hundreds of high-velocity particles, while stillness tightens the "aura".

## 📸 Usage & Controls

### Installation
```bash
pip install -r requirements.txt
```

### Running the App
```bash
python main.py
```

### Controls
To get the best visual experience, make sure you have good lighting on yourself so the AI can track you.
- **Action**: Swipe your hands fast to "ignite" the stardust and create glowing neon arcs.
- `b` Key: Toggle between the **Pitch-Black Background** (for maximum neon contrast) and the **Darkened Webcam Background** (to show the real-time AI magic).
- `q` Key: Quit the application.

---

### Tips for Content Creation (LinkedIn / Socials)
If you are recording this for a showcase:
1. Start the video with the screen completely black (`b` toggled off).
2. Walk into the frame and "swipe" your hand to "ignite" the particles. That transition from nothing to a "Digital Ghost" is a scroll-stopper!
3. Toggle the background (`b`) mid-video to reveal your silhouette and prove it's a live AI system.

*#ComputerVision #GenerativeArt #Python #AI #MediaPipe*
