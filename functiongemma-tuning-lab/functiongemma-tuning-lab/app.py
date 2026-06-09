from ui import build_interface

if __name__ == "__main__":
    # Build and Launch UI
    # Note: Engine creation is now handled per-session inside build_interface
    demo = build_interface()
    print("Starting Gradio App with Multi-User Support...")
    demo.queue() # Enable queueing for concurrent request handling
    demo.launch()
