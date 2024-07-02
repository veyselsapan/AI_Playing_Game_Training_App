# src/main.py

import tkinter as tk
from gui import BreakoutAIApp

def main():
    root = tk.Tk()
    app = BreakoutAIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
