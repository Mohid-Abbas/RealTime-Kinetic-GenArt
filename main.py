from src.kinetic_ghost import KineticGhostApp

if __name__ == "__main__":
    try:
        app = KineticGhostApp()
        app.run()
    except Exception as e:
        print(f"Error starting KineticGhost: {e}")
