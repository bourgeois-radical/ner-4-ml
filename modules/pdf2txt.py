import pathlib
import sys
import pickle

from typing import List, Tuple
from pypdf import PdfReader
from pathlib import Path

# from preprocessing import DefaultPreprocess


def txt_from_pdf(src_pdf_path: str, dst_txt_path: str, preprocess: bool = False) -> str:
    """Extracts text from PDF files (one column structure as well as two column structure)
    and saves it as a .txt file

    Parameters
    ----------
    src_pdf_path: str
        Path
    dst_txt_path: str
    preprocess: bool

    Returns
        str: a massage, which says, whether something went wrong or not.

    -------

    """
    # transform str path to WindowsPath
    dst_txt_path = Path(dst_txt_path)

    # get the path of each and every pdf file
    pointer_pdf_paths = Path(src_pdf_path).glob('**/*')
    all_pdf_paths = [x for x in pointer_pdf_paths if x.is_file()]

    # extract text from pdf
    full_pdf_text = ''
    for single_pdf_path in all_pdf_paths:

        # some pdfs may appear to be encrypted
        if PdfReader(single_pdf_path).is_encrypted:
            print(f'The following file is encrypted: {single_pdf_path}')
            not_all_were_converted = True
            continue
        else:
            reader = PdfReader(single_pdf_path)
            number_of_pages = len(reader.pages)

        for page_number in range(number_of_pages):
            page = reader.pages[page_number]
            text = page.extract_text()
            full_pdf_text = full_pdf_text + ' ' + text

        # if preprocess:
        # don't need preprocessing so far, since it depends on a model.

        # writing to the .txt file
        dst = str(dst_txt_path / single_pdf_path.name.replace('pdf', 'txt'))
        with open(dst, 'w', encoding='utf-8') as txt_file:
            txt_file.write(full_pdf_text)

    if not_all_were_converted:
        return 'NOT all pdf files have been successfully converted to txt files.'
    else:
        return 'All pdf files have been successfully converted to txt files.'
