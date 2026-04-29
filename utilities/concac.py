"""
Concatenate all PDF files in a folder into a single PDF
"""

from pypdf import PdfWriter
from pathlib import Path
import os

def concatenate_pdfs(folder_path, output_filename="merged_music.pdf"):
    """
    Concatenate all PDF files in a folder into a single PDF.
    
    Args:
        folder_path: Path to folder containing PDF files
        output_filename: Name of the output merged PDF file
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return False
    
    # Find all PDF files in the folder
    pdf_files = sorted(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder}")
        return False
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    # Create PDF writer
    writer = PdfWriter()
    
    # Add each PDF to the writer
    total_pages = 0
    for pdf_file in pdf_files:
        try:
            print(f"Adding: {pdf_file.name}")
            writer.append(str(pdf_file))
            total_pages += len(writer.pages)
        except Exception as e:
            print(f"  Error adding {pdf_file.name}: {e}")
            continue
    
    # Write the merged PDF
    output_path = folder / output_filename
    try:
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        print(f"\nSuccess! Merged PDF created:")
        print(f"  {output_path}")
        print(f"  Total pages: {len(writer.pages)}")
        return True
    except Exception as e:
        print(f"Error writing merged PDF: {e}")
        return False


if __name__ == "__main__":
    # Folder containing PDF files
    folder_path = r"C:\Users\tim_e\Downloads\music_end_feb"
    
    # Output filename
    output_filename = "merged_music_feb.pdf"
    
    print("=" * 70)
    print("PDF Concatenation Tool")
    print("=" * 70)
    print(f"Source folder: {folder_path}")
    print(f"Output file: {output_filename}")
    print("=" * 70)
    print()
    
    # Run concatenation
    success = concatenate_pdfs(folder_path, output_filename)
    
    if success:
        print("\n" + "=" * 70)
        print("PDF concatenation complete!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("PDF concatenation failed")
        print("=" * 70)
