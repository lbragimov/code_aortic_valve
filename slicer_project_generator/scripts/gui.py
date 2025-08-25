# scripts/gui.py
import tkinter as tk
from tkinter import simpledialog

class CaseSelector:
    def __init__(self, cases):
        self.cases = cases
        self.selected = None

    def run(self):
        root = tk.Tk()
        root.withdraw()  # скрыть главное окно

        self.selected = simpledialog.askstring(
            "Выбор кейса",
            "Введите имя кейса:\n" + "\n".join(self.cases)
        )

        root.destroy()
        return self.selected