import pandas as pd

def load_file(file_path):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("File format not supported. Please provide a CSV or Excel file.")
    except Exception as e:
        raise ValueError(f"Failed to load file: {e}")

def chat_with_bot(df):
    print("Hello! I am your data chatbot. Ask me about the data.")
    print("You can type 'exit' to end the conversation.")
    
    columns = {col.lower(): col for col in df.columns if not col.lower().startswith('unnamed')}  # Map lowercase column names to original names
    
    while True:
        user_input = input("You: ").strip().lower()
        
        if user_input == 'exit':
            print("Goodbye!")
            break
        
        if user_input in columns:
            col_name = columns[user_input]
            print(f"Here are the first 5 values in the column '{col_name}':")
            print(df[col_name].head())
        else:
            print("I didn't understand that. Please ask about a specific column in the data.")

def main():
    file_path = input("Please provide the path to your CSV or Excel file: ").strip().strip('"')
    try:
        df = load_file(file_path)
        valid_columns = [col for col in df.columns if not col.startswith('Unnamed')]
        df = df[valid_columns]  # Filter the dataframe to include only valid columns
        print("File loaded successfully!")
        print(f"Columns available: {', '.join(valid_columns)}")
        chat_with_bot(df)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
