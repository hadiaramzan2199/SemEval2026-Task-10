import json
import re

def clean_text_safe(text):
    """
    Clean text by converting to lowercase and normalizing quotes
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    text = text.lower()
    
    # Replace fancy unicode quotes with ASCII
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace('"', '"').replace('"', '"')
    
    return text


def load_and_preprocess_data(filepath):
    """
    Load JSONL data and apply preprocessing
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of preprocessed data items
    """
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            item["text"] = clean_text_safe(item["text"])
            data.append(item)
    
    print(f"Loaded {len(data)} examples from {filepath}")
    return data


def rehydrate_data(input_path, output_path):
    """
    Rehydrate redacted data using external script
    
    Note: This function assumes rehydrate_data.py is available
    """
    import subprocess
    
    cmd = [
        "python", "rehydrate_data.py",
        "--input", input_path,
        "--output", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Data rehydrated and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error rehydrating data: {e}")
        raise
    except FileNotFoundError:
        print("rehydrate_data.py not found. Please download it from:")
        print("https://raw.githubusercontent.com/hide-ous/semeval26_task10_starter_pack/main/rehydrate_data.py")


if __name__ == "__main__":
    # Example usage
    data = load_and_preprocess_data("train_rehydrated.jsonl")
    print(f"Sample text: {data[0]['text'][:100]}...")
    print(f"Number of markers in first example: {len(data[0].get('markers', []))}")