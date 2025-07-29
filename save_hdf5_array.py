from astropy.io import fits
import pandas as pd
import os
import argparse

def extract_lightcurves(file_path, start_index, end_index, output_dir):
    '''
    Extracts lightcurves from rome crossmatch fits file from start_index to end_index
    ans saves them as a csv

    file_path:   file path to fits file
    start_index: first index within fits file you want the lightcurve for
    end_index:   last index within fits file you want the lightcurve for
    output_dir:  directory to save lightcurve csv files

    To use, type in terminal (in the directory where this code is placed):
    python save_hdf5_array.py <file_path to fits file> --start <start_index> 
        --end <end_index> --outdir <directory where you want lightcurves to go>
    '''
    os.makedirs(output_dir, exist_ok=True)
    
    # load data & extract lightcurves
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        total_rows = len(data)
        end_index = min(end_index, total_rows)
        
        print(f"Extracting lightcurves {start_index} to {end_index-1} from {total_rows} total rows")
        
        for i in range(start_index, end_index):
            # convert to dataframe and save as csv
            row_dict = {col: [data[i][col]] for col in data.columns.names}
            df = pd.DataFrame(row_dict)
            df.to_csv(f"{output_dir}/lightcurve_{i:06d}.csv", index=False)
            
            if (i - start_index + 1) % 50 == 0:
                print(f"Processed {i - start_index + 1}/{end_index - start_index} lightcurves")
        
        print(f"Extracted {end_index - start_index} lightcurves to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract lightcurves from FITS file")
    parser.add_argument("file_path", help="Path to FITS file")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--end", type=int, default=100, help="End index (default: 100)")
    parser.add_argument("--outdir", default="lightcurves", help="Output directory (default: lightcurves)")
    
    args = parser.parse_args()
    extract_lightcurves(args.file_path, args.start, args.end, args.outdir)