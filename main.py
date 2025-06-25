import os

def main():
    """Runs the Streamlit app."""
    os.system("/usr/bin/python3 -m streamlit run app.py --server.headless true")

if __name__ == "__main__":
    main()
