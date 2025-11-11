#!/usr/bin/env python3
"""
Test script for CSV Formatter agent
"""

import sys
from pathlib import Path
import os

# Add paths for imports
current_dir = Path(__file__).parent
utils_path = current_dir / "utils"
sys.path.insert(0, str(utils_path))
sys.path.insert(0, str(current_dir))

def test_csv_formatter():
    """Test the CSV Formatter with sample data"""
    
    # Sample complex CSV data (similar to stat.gov.az format)
    sample_csv_data = """
,,,,,,,,,,,,,,,,,,,,,,,,,
,"2.1. Kənd təsərrüfatı məhsulları istehsalının fiziki həcm indeksi, müqayisəli qiymətlərlə, əvvəlki ilə nisbətən",,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,
,,2003,,,2004,,,2005,,,2006,,,2007,,,2008,,,2009,,,2010,,
,,Cəmi,"bitki-
çilik","heyvan-
darlıq",Cəmi,"bitki-
çilik","heyvan-
darlıq",Cəmi,"bitki-
çilik","heyvan-
darlıq",Cəmi,"bitki-
çilik","heyvan-
darlıq",Cəmi,"bitki-
çilik","heyvan-
darlıq",Cəmi,"bitki-
çilik","heyvan-
darlıq",Cəmi,"bitki-
çilik","heyvan-
darlıq",Cəmi,"bitki-
çilik","heyvan-
darlıq"
,Azərbaycan Respublikası,105.6,105.3,106.2,104.6,104.0,105.7,107.5,110.5,103.6,100.9,99.2,102.9,104.0,102.4,106.1,106.1,107.2,104.1,103.5,104.0,102.8,97.8,91.1,106.1
,Bakı şəhəri ,101.1,99.5,104.2,103.2,100.0,107.1,98.3,97.1,99.7,182.8,274.2,103.0,101.6,102.2,99.9,116.8,112.4,130.7,101.0,89.6,117.9,77.8,69.7,90.7
,Naxçıvan Muxtar Respublikası,103.5,104.6,100.9,103.6,100.9,107.7,112.5,118.6,105.1,104.7,104.2,105.4,102.0,100.0,103.7,111.5,116.6,104.9,113.4,118.0,105.1,113.3,116.0,108.3
,Naxçıvan şəhəri,82.3,121.7,65.5,116.8,98.5,127.1,102.1,122.4,101.4,107.4,107.4,105.1,106.9,107.0,103.4,104.1,105.0,104.1,298.3,559.9,166.5,119.1,111.8,119.9
,Babək rayonu,105.9,109.4,100.0,105.3,105.3,105.3,109.9,109.9,111.7,104.7,104.4,105.3,101.5,101.5,104.3,112.8,116.5,105.2,97.4,97.2,97.7,114.0,113.9,114.0
,Culfa rayonu,100.9,101.2,100.1,104.4,100.3,111.5,118.8,131.2,103.1,101.7,100.0,104.7,102.7,102.3,103.4,112.7,118.5,104.7,114.4,119.3,106.1,112.9,116.6,105.5
,Kəngərli rayonu,-,-,-,88.9,77.8,103.6,119.4,137.9,104.5,105.5,106.0,104.0,101.9,100.5,103.8,110.0,116.4,103.7,115.1,121.1,104.5,113.4,118.6,106.0
,Ordubad rayonu,103.7,102.0,105.1,110.5,101.3,118.1,108.9,113.1,105.6,107.5,110.5,104.4,103.5,103.5,103.5,111.6,117.3,106.4,114.1,120.0,107.4,112.1,114.2,109.7
,Sədərək rayonu,87.1,87.1,86.7,101.3,91.4,150.1,143.4,162.2,104.2,104.0,103.9,104.6,101.1,100.5,103.5,110.1,112.1,104.5,111.8,114.2,103.7,110.1,111.1,106.4
,Şahbuz rayonu,105.4,106.1,105.2,118.0,154.7,108.8,129.4,201.1,107.5,105.7,106.6,105.1,102.1,100.7,103.1,108.7,112.9,106.5,120.3,143.4,106.5,115.3,124.6,108.3
,Şərur rayonu,101.5,102.3,101.0,102.0,101.0,103.6,103.9,103.5,104.4,105.0,104.1,106.5,101.7,100.3,104.0,111.7,117.5,104.2,117.2,124.6,104.9,112.4,115.9,106.6
,Abşeron-Xızı iqtisadi rayonu,130.0,110.3,135.2,108.9,109.3,108.9,103.5,90.7,104.8,79.6,112.3,77.7,123.0,90.8,126.7,105.5,94.1,106.4,104.3,101.6,104.5,93.6,115.3,90.4
,Sumqayıt şəhəri,96.5,100.6,93.7,108.2,111.5,106.2,102.4,116.9,91.8,101.7,100.5,102.6,97.2,94.7,99.9,72.2,36.2,128.7,111.6,103.2,115.8,122.4,132.8,118.4
,Abşeron rayonu,130.4,113.9,153.1,108.3,109.9,108.2,102.6,79.3,104.0,85.7,143.1,83.7,105.0,90.9,106.2,101.8,112.0,101.2,106.3,103.0,106.5,82.5,117.5,79.7
,Xızı rayonu,130.8,107.7,151.1,110.8,108.2,111.3,107.7,98.8,108.9,61.3,86.1,58.7,194.7,89.0,214.8,114.8,86.2,116.6,100.7,97.3,100.9,116.0,105.7,116.1
,Dağlıq Şirvan iqtisadi rayonu,103.2,106.5,100.1,102.5,103.3,101.6,100.8,98.0,103.7,100.1,99.1,101.0,102.1,103.2,101.2,107.0,112.5,101.1,103.5,105.4,101.6,96.2,75.5,112.8
,Ağsu rayonu,104.5,117.4,92.8,109.7,117.0,100.2,106.1,101.5,112.0,100.7,100.0,101.5,105.9,110.7,100.6,105.7,109.8,100.2,99.8,99.4,100.0,90.6,72.7,104.5
,İsmayıllı rayonu,106.6,111.8,100.9,97.5,93.8,103.1,107.0,112.7,99.9,97.5,95.3,100.1,96.1,92.4,100.7,106.9,111.3,100.9,98.7,98.1,99.7,101.8,80.3,127.8
,Qobustan rayonu,102.7,96.5,108.6,104.9,106.0,104.0,79.9,48.6,101.8,116.6,164.1,102.2,101.1,97.6,102.9,106.8,113.9,102.8,114.8,126.6,106.8,88.7,66.9,101.7
,Şamaxı rayonu,96.9,92.6,100.6,99.1,99.3,99.0,101.4,102.0,101.1,90.8,76.0,100.5,107.7,119.6,100.7,109.2,119.9,100.8,107.8,117.0,100.6,102.6,79.4,122.9
,Gəncə-Daşkəsən iqtisadi rayonu,103.6,112.7,102.0,105.2,112.3,104.7,113.9,118.3,107.4,103.2,102.0,103.9,110.8,120.4,104.0,112.5,115.7,102.8,108.6,108.9,104.5,98.7,90.5,108.3
,Gəncə şəhəri,104.9,111.2,103.2,112.4,99.0,122.0,120.9,164.7,94.1,98.6,100.1,97.2,138.8,169.6,107.4,193.0,272.5,66.7,56.4,47.1,98.7,105.3,115.2,74.1
,Naftalan şəhəri,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-
"""

    try:
        # Test without API key first (will fail gracefully)
        print("Testing CSV Formatter...")
        print("\nOriginal CSV Data:")
        print("-" * 50)
        print(sample_csv_data.strip())
        
        # Import pandas and io first
        import pandas as pd
        import io
        
        # Import the utilities with absolute imports
        from utils.data_processing import DataProcessor
        
        # Test data processing utilities
        print("\n\nTesting DataProcessor utilities...")
        
        # Parse the raw CSV to check formatting needs
        df = pd.read_csv(io.StringIO(sample_csv_data))
        needs_formatting = DataProcessor.detect_if_needs_formatting(df)
        
        print(f"Original DataFrame shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        print(f"Needs formatting: {needs_formatting}")
        
        # Show why it needs formatting
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        print(f"Unnamed columns: {len(unnamed_cols)}")
        
        # Test if we have an API key in environment
        api_key = os.getenv('GOOGLE_API_KEY', '')
        
        if api_key:
            print(f"\nAPI key found, testing CSV formatting...")
            
            # Import and initialize formatter
            from utils.ai_agents import CSVFormatter
            from utils.data_processing import DataProcessor
            formatter = CSVFormatter(api_key=api_key)
            
            # Format the CSV (now returns Dict[str, str])
            tables_dict = formatter.format_csv_data(sample_csv_data)
            
            print(f"\nFormatted CSV Tables (detected {len(tables_dict)} table(s)):")
            print("-" * 50)
            
            # Parse tables into DataFrames
            dataframes_dict = DataProcessor.parse_multiple_formatted_csv_strings(tables_dict)
            
            for i, (table_name, csv_data) in enumerate(tables_dict.items(), 1):
                print(f"\n=== TABLE {i}: {table_name} ===")
                print(csv_data)
                
                if table_name in dataframes_dict:
                    df_formatted = dataframes_dict[table_name]
                    print(f"\nDataFrame shape: {df_formatted.shape}")
                    print(f"Columns: {list(df_formatted.columns)}")
                    print("\nFirst 3 rows:")
                    print(df_formatted.head(3))
                else:
                    print("Could not parse this table into DataFrame")
                print("-" * 50)
            
        else:
            print("\nNo API key found in environment. Set GOOGLE_API_KEY to test AI formatting.")
            print("Testing completed with local utilities only.")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_csv_formatter()
