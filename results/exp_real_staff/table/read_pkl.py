import pickle
import os
import sys

def read_pickle_file(file_path: str):
    """
    Liest eine .pkl-Datei und gibt den Inhalt zurück.

    Args:
        file_path (str): Der Pfad zur .pkl-Datei.

    Returns:
        any: Der Inhalt der .pkl-Datei oder None, wenn ein Fehler auftritt.
    """
    try:
        print(f"Versuche, Datei zu lesen: {file_path}")  # Debug-Ausgabe
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Datei erfolgreich gelesen: {file_path}")  # Debug-Ausgabe
        return data
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden unter {file_path}")
        return None
    except pickle.UnpicklingError:
        print(f"Fehler: Ungültige .pkl-Datei unter {file_path}")
        return None
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        return None

def process_pickle_files(directory):
    """
    Verarbeitet alle .pkl-Dateien in einem Verzeichnis.

    Args:
        directory (str): Der Pfad zum Verzeichnis, das .pkl-Dateien enthält.
    """
    print(f"Starte process_pickle_files mit Verzeichnis: {directory}")  # Debug-Ausgabe
    if not os.path.exists(directory):
        print(f"Fehler: Verzeichnis nicht gefunden unter {directory}")
        return

    if not os.path.isdir(directory):
        print(f"Fehler: {directory} ist kein Verzeichnis.")
        return

    try:
        files = os.listdir(directory)
    except OSError as e:
        print(f"Fehler beim Auflisten des Verzeichnisses {directory}: {e}")
        return

    if not files:
        print(f"Warnung: Das Verzeichnis {directory} ist leer.")
        return

    for filename in files:
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                print(f"Warnung: {file_path} ist keine Datei.")
                continue
            if not os.access(file_path, os.R_OK):
                print(f"Warnung: Keine Leseberechtigung für {file_path}.")
                continue

            data = read_pickle_file(file_path)
            if data is not None:
                print(f"Inhalt von {filename}:")
                print(data)  # Hier kannst du die Daten weiterverarbeiten
                print("-" * 20)
    print("process_pickle_files beendet")  # Debug-Ausgabe

if __name__ == "__main__":
    print("Starte Skript")  # Debug-Ausgabe
    current_directory = "results/exp_real_staff/table"  # Aktuelles Verzeichnis abrufen
    process_pickle_files(current_directory)
    print("Skript beendet")  # Debug-Ausgabe