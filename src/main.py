# This is a sample Python script.
# from datetime import datetime
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import Counter
from datetime import date
from pathlib import Path

from Experimenter import Experimenter, DatasetInputType

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

core_path = Path(__file__).parent.resolve() / "../../per-year-test/"
core_path = Path(__file__).parent.resolve() / "../../test-around/"
#core_path = Path(__file__).parent.resolve() / "../../specific-test/"
core_path = Path(__file__).parent.resolve() / "../../per-year/"
core_path = Path(__file__).parent.resolve() / "../../FINAL-TEST/"
print(core_path)

final_paper_directory = str(core_path)
testPlot = Experimenter(str(final_paper_directory), "AIIDE papers")
#testPlot.load_all_papers_pdf(str(final_paper_directory))
testPlot.load_compressed_view() # After we have actually loaded all the raw data from html, it is easy and quick to load comprresed
testPlot.calculate_metrics_references(column_name="references")

#testPlot.search_keywords_per_paper(keyword="pcg", sentences_to_search=["PCG", "procedural content generation", "procedurally generated content", "content generation", "procedural generation", "procedurally generating"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="mixed-initiative", sentences_to_search=["mixed-initiative", "mixedinitiative", "mixed initiative"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="CAD", sentences_to_search=["computer-aided game design", "computer-aided design", "computer aided design", "computer aided game design", "computer-aided", "computer aided", "computer aid", "computer-aid"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="computer-aided", sentences_to_search=["computer-aided", "computer aided", "computer aid", "computer-aid"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="human-ai", sentences_to_search=["human-ai", "humanai"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="human-ai-collaboration-inter", sentences_to_search=["human-ai collaboration", "humanai collaboration", "humanai interaction", "human-ai interaction"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="computational-support", sentences_to_search=["computational support", "computationally support", "computationally supported"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="design-assistant", sentences_to_search=["design assistant", "designer assistant", "design-assistant", "intelligent design assistant", "intelligent designer assistant"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="ai-assisted", sentences_to_search=["ai assisted", "ai-assisted", "ai assistant", "ai-assistant"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="co-design", sentences_to_search=["co-design", "codesign", "co-designing", "codesigning", "codesigner", "co-designer"], minimum_length=1, top_k=10)
testPlot.search_keywords_per_paper(keyword="co-creative", sentences_to_search=["co-creative", "co-creativity", "cocreative", "cocreativity"], minimum_length=1, top_k=10)

testPlot.save_aggregated_encountered_papers()
testPlot.extract_statistics("keyword", testPlot.aggregated_encountered_papers, False)
# # Extract most used sentences in all papers and all sections (you can set the minimum length)
# testPlot.extract_sentences(column_name="nothing")
# testPlot.extract_sentences(column_name="nothing", minimum_length=3)
# testPlot.extract_sentences(column_name="nothing", minimum_length=6)
#
# testPlot.extract_sentences_per_paper(column_name="not important")
# testPlot.extract_sentences_per_paper(column_name="not important", minimum_length=3)
# testPlot.extract_sentences_per_paper(column_name="not important", minimum_length=6)
# #
# testPlot.search_sentences_per_paper(sentences_to_search=["contribution", "three research questions",
# "research question", "our knowledge"], minimum_length=3, top_k=10)
# testPlot.search_sentences_per_paper(sentences_to_search=["contribution", "three research questions",
# "research question", "our knowledge"], minimum_length=6, top_k=10)
# testPlot.search_sentences_per_paper(sentences_to_search=["contribution", "three research questions",
# "research question", "our knowledge"], minimum_length=9, top_k=10)
# #
# # # testPlot.calculate_ngram_unique(column_name="no_matter", ngram=10, all_ngrams_until=True, top_k=5)
# testPlot.calculate_ngram(column_name="intro", ngram=10, top_k=25)
# testPlot.calculate_ngram_per_paper(column_name="no_matter", ngram=10, all_ngrams_until=True, top_k=25)

# core_path = Path(__file__).parent.resolve() / "../data/"
# print(core_path)
#
# validation_path = Path(__file__).parent.resolve() / "../data/validation-data/"
# print(validation_path)
#
# paper_directory = str(core_path) + "/"
# paper_directory = str(core_path) + "/compressed_view/"
#
# testPlot = Experimenter(paper_directory, "CHI 2023 papers")
#
# # First run the code with this uncommented. Then comment the code, and run testPlot.load_compressed_view()
# # testPlot.load_all_original_papers_data() # load the raw info from the website
# testPlot.load_compressed_view() # After we have actually loaded all the raw data from html, it is easy and quick to load comprresed
# testPlot.calculate_papers_metrics(column_name="author_names")
# testPlot.calculate_metrics_for_sessions(column_name="sessions")
# testPlot.calculate_metrics_sections(column_name="sections")
# testPlot.calculate_metrics_references(column_name="references")
#
# # Uncomment to create the expected dataset used in the paper for alt.chi
# testPlot.create_limited_dataset_text_references(DatasetInputType.NO_REFERENCES_SECTION_PARTS, token_limit=2048,
#                                                 section_token_limit=2048) # Create a Dataset

# import fitz
# import pymupdf
# import re
#
#
# def starts_with_figure_number(text):
#     # Define the regex pattern
#     pattern = r"^Figure \d+"
#
#     # Search for the pattern in the text
#     match = re.match(pattern, text)
#
#     # Return True if a match is found, else False
#     return bool(match)
#
# def starts_with_table_number(text):
#     # Define the regex pattern
#     pattern = r"^Table \d+"
#
#     # Search for the pattern in the text
#     match = re.match(pattern, text)
#
#     # Return True if a match is found, else False
#     return bool(match)
#
#
# def combine_positions(table_positions, min_threshold=30):
#     combined_positions = {"x0": [], "x1": [], "y0": [], "y1": []}
#
#     if not table_positions["x0"]:
#         return table_positions
#
#     # Initial positions
#     current_x0 = table_positions["x0"][0]
#     current_x1 = table_positions["x1"][0]
#     current_y0 = table_positions["y0"][0]
#     current_y1 = table_positions["y1"][0]
#
#     for i in range(1, len(table_positions["x0"])):
#         next_x0 = table_positions["x0"][i]
#         next_x1 = table_positions["x1"][i]
#         next_y0 = table_positions["y0"][i]
#         next_y1 = table_positions["y1"][i]
#
#         # Check if the difference between current y1 and next y0 is within the threshold
#         if min_threshold <= next_y0 - current_y1:
#             # Save the current combined positions
#             combined_positions["x0"].append(current_x0)
#             combined_positions["x1"].append(current_x1)
#             combined_positions["y0"].append(current_y0)
#             combined_positions["y1"].append(current_y1)
#
#             # Move to the next position
#             current_x0 = next_x0
#             current_x1 = next_x1
#             current_y0 = next_y0
#             current_y1 = next_y1
#         else:
#             # Combine the positions
#             current_x1 = max(current_x1, next_x1)
#             current_y1 = next_y1
#
#     # Append the last set of combined positions
#     combined_positions["x0"].append(current_x0)
#     combined_positions["x1"].append(current_x1)
#     combined_positions["y0"].append(current_y0)
#     combined_positions["y1"].append(current_y1)
#
#     return combined_positions
#
# def check_if_header_footer(bounds, page_bounds, page_bound_perc=0.08):
#     if 66 < bounds[1] < page_bounds[3] - (page_bounds[3] * page_bound_perc):
#         return False
#     else:
#         return True
#
# def extract_section_names(file_path):
#     pdf_document = pymupdf.open(file_path)
#     print(pdf_document.get_toc())
#
#     document_toc = pdf_document.get_toc()
#
#     # Extracting the second element where the first element is 1 (1 means that it is a section!)
#     extracted_sections = [sublist[1] for sublist in document_toc if sublist[0] == 1]
#     print(extracted_sections)
#     return extracted_sections
#
# def extract_text_from_pdf(file_path):
#
#     extra_margin_tables = 10
#
#     # Open the PDF file
#     pdf_document = pymupdf.open(file_path)
#
#     print(pdf_document.page_count)
#     print(pdf_document.metadata["author"])
#     print(pdf_document.metadata["keywords"])
#     print(pdf_document.metadata["title"])
#
#     print(pdf_document.get_toc())
#
#     extracted_text = []
#
#     html_page = pdf_document[0].get_text("html")
#     xml_page = pdf_document[0].get_text("xml")
#
#
#
#     for page_num in range(len(pdf_document)):
#         print("PAGE NUMBER::: " + str(page_num))
#         page = pdf_document[page_num]
#         text = page.get_text("blocks")
#         table_positions = {"x0": [], "x1": [], "y0": [], "y1": []}
#         page_bounds = page.bound()
#
#         tabs = page.find_tables()
#         print(f"{len(tabs.tables)} table(s) on {page}")
#
#         for tab in tabs:
#             table_positions["x0"].append(tab.bbox[0] - extra_margin_tables)
#             table_positions["x1"].append(tab.bbox[2] + extra_margin_tables)
#             table_positions["y0"].append(tab.bbox[1] - extra_margin_tables)
#             table_positions["y1"].append(tab.bbox[3] + extra_margin_tables)
#
#             for line in tab.extract():
#                 print(line)
#
#         table_positions = combine_positions(table_positions)
#
#         for block in text:
#             # block is a tuple: (x0, y0, x1, y1, "text", block_no, block_type)
#             add_text = True
#             for table_numbers in range(len(table_positions['x0'])):
#                 # The text is within the height bounds of the table (so... possible text within a table)
#                 if block[1] > table_positions['y0'][table_numbers] and block[1] < table_positions['y1'][table_numbers]:
#
#                     # Now we check if the text is within the width bounds of the table
#                     if block[0] < table_positions['x1'][table_numbers]:
#                         add_text = False
#                         break
#
#             if not add_text:
#                 continue
#             elif starts_with_figure_number(block[4]) or starts_with_table_number(block[4]):
#                 continue
#             elif (block[4].startswith("Permission to make") or block[4].startswith("ACM Reference Format")
#                   or block[4].startswith("Authors’ addresses:") or block[4].startswith("Publication rights licensed")):
#                 continue
#             elif check_if_header_footer(block, page_bounds):
#                 continue
#
#             extracted_text.append(block[4])
#             print(block[4])
#             if block[6] != 0:
#                 print(block[6])
#
#     return extracted_text
#
# section_names = extract_section_names(paper_directory)
# text_data = extract_text_from_pdf(paper_directory)
#
# for text in text_data:
#     print(text)
#
#
# from pdfminer.high_level import extract_text
#
# def extract_text_from_pdf(pdf_path):
#     text = extract_text(pdf_path)
#     paragraphs = text.split('\n')  # Assuming paragraphs are separated by double line breaks
#     return paragraphs
#
# pdf_path = paper_directory  # Change this to your PDF file path
# paragraphs = extract_text_from_pdf(pdf_path)
# for paragraph in paragraphs:
#     print(paragraph)
#
#
# from pdfminer.high_level import extract_pages
# from pdfminer.layout import LTTextBox
#
# def extract_text_from_LTTextBoxes(pdf_path):
#     text = ""
#     for page_layout in extract_pages(pdf_path):
#         for element in page_layout:
#             if isinstance(element, LTTextBox):
#                 print(":::::")
#                 print(element.get_text())
#                 print("-----------------------")
#                 text += element.get_text()
#     return text
#
# # Replace 'your_pdf_file.pdf' with the path to your PDF file
# pdf_path = paper_directory
# extracted_text = extract_text_from_LTTextBoxes(pdf_path)
# print(extracted_text)
#
# import fitz
#
# doc = fitz.open(paper_directory)
# for page in doc:
#     tabs = page.find_tables()
#     if tabs.tables:
#         print(tabs[0].extract())
#
# import camelot
#
# # Extract tables from the PDF
# tables = camelot.read_pdf(paper_directory, flavor='stream', pages='all')
#
# # Get the bounding boxes of the tables, organized by page number
# table_areas_by_page = {}
# for table in tables:
#     page_num = table.page - 1  # Camelot's pages are 1-indexed, Fitz uses 0-indexed
#     bbox = table._bbox
#     if page_num not in table_areas_by_page:
#         table_areas_by_page[page_num] = []
#     table_areas_by_page[page_num].append(bbox)
#
# import fitz  # PyMuPDF
#
# def convert_bbox(bbox, page_height):
#     """
#     Convert Camelot bounding box to Fitz bounding box.
#     Camelot: (x1, y1, x2, y2) with (0,0) at bottom-left.
#     Fitz: (x0, y0, x1, y1) with (0,0) at top-left.
#     """
#     x0, y0, x1, y1 = bbox
#     y0_converted = page_height - y1  # Convert y-coordinates
#     y1_converted = page_height - y0
#     return fitz.Rect(x0, y0_converted, x1, y1_converted)
#
# def mask_table_areas(pdf_path, table_areas_by_page):
#     doc = fitz.open(pdf_path)
#     for page_num in table_areas_by_page:
#         page = doc.load_page(page_num)
#         page_height = page.rect.height  # Get the height of the page
#         for bbox in table_areas_by_page[page_num]:
#             rect = convert_bbox(bbox, page_height)  # Convert bbox to fitz.Rect format
#             # Draw a white rectangle over the table area
#             page.draw_rect(rect, color=(1, 1, 1), fill=True)
#     output_path = str(validation_path) + "/modified_" + "2658537.2658681.pdf"
#
#     doc.save(output_path)
#     doc.close()
#     return output_path
#
# masked_pdf_path = mask_table_areas(paper_directory, table_areas_by_page)
# print(f"Modified PDF saved as: {masked_pdf_path}")
#
#
# import pdfplumber
#
#
# def not_within_bboxes(obj):
#     """Check if the object is in any of the table's bbox."""
#
#     def obj_in_bbox(_bbox):
#         """See https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404"""
#         v_mid = (obj["top"] + obj["bottom"]) / 2
#         h_mid = (obj["x0"] + obj["x1"]) / 2
#         x0, top, x1, bottom = _bbox
#         return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
#
#     return not any(obj_in_bbox(__bbox) for __bbox in bboxes)
#
#
# with pdfplumber.open(paper_directory) as pdf:
#     for page in pdf.pages:
#         print("\n\n\n\n\nAll text:")
#         print(page.extract_text())
#
#         # Get the bounding boxes of the tables on the page.
#         bboxes = [
#             table.bbox
#             for table in page.find_tables(
#                 table_settings={
#                     "vertical_strategy": "explicit",
#                     "horizontal_strategy": "explicit",
#                     "explicit_vertical_lines": page.curves + page.edges,
#                     "explicit_horizontal_lines": page.curves + page.edges,
#                 }
#             )
#         ]
#
#         print("\n\n\n\n\nText outside the tables:")
#         print(page.filter(not_within_bboxes).extract_text())
