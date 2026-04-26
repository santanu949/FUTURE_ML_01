import typer
import uvicorn
from src.viz.dashboard import run_dashboard
from src.data.processor import DataGenerator

app = typer.Typer()

@app.command()
def dashboard():
    """Launch the Streamlit interactive dashboard."""
    import os
    print("Launching Streamlit dashboard...")
    os.system("streamlit run src/viz/dashboard.py")

@app.command()
def serve(port: int = 8000):
    """Start the FastAPI REST server."""
    print(f"Starting API server on port {port}...")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=port, reload=True)

@app.command()
def generate(periods: int = 365, output: str = "sample_data.csv"):
    """Generate synthetic sales data for testing."""
    df = DataGenerator.generate_sales_data(periods=periods)
    df.to_csv(output, index=False)
    print(f"Generated {periods} days of data and saved to {output}")

if __name__ == "__main__":
    app()
