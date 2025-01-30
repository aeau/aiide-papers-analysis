from collections import Counter
from datetime import date
from locale import atoi

import re

import chardet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os as os
import os.path
from os import path
from numpy import genfromtxt
from os import listdir
import copy
import enum
import math
from enum import Enum
import matplotlib as mpl
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation

from matplotlib.colors import from_levels_and_colors
from matplotlib.collections import LineCollection
from sympy import false

from src.CollectedPapersGeneral import CollectedPapersGeneral
from src.EncounteredPapersGeneral import EncounteredPapersGeneral
from src.Paper import Paper
from src.EncounteredPapers import EncounteredPapers
from src.DatasetLLMPapers import *
import string

import pymupdf

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTLine, LAParams, LTAnno

from Experimenter import DatasetInputType

class ExperimenterFDG:

    def __init__(self, papers_directory, experiment_name="itiswhatitis"):
        print("experimentorFDG")
        self.experiment_name = experiment_name
        self.papers_directory = papers_directory
        self.papers = {}
        self.compressed_papers = {}

        self.column_names = ["article-authors", "article-fullauthors", "article-headers", "article-sections", "article-ind-bib"]
        self.variable_names = ["author", "author_institution", "section_title", "section", "reference"]
        self.max_sections = 50
        self.max_sections_paper_name = ""

        self.collected_papers_general = CollectedPapersGeneral("chi23", self.max_sections)

        self.ignore_keywords_analysis = ["publication rights", "acm", "licensed under a creative commons",
                           "human factors in computing", "licensed", "license", "Licensed", "profit", "proceedings", "sigchi",
                           "commercial", "citation", "classroom"]


        self.encountered_papers = {}
        self.aggregated_encountered_papers = {}
        self.encountered_papers_general = EncounteredPapersGeneral("aiide", 50)

    def get_csv_files(self, directory):
        filenames = listdir(directory)
        csv_files = [filename for filename in filenames if filename.endswith(".csv")]

        return csv_files

    def get_files(self, directory, extension : str):
        filenames = listdir(directory)
        files = [filename for filename in filenames if filename.endswith(extension) and not filename.startswith("~")]

        return files

    def extract_sessions(self, df):
        """Extract session name for the papers and their name
        Creates the Paper object with the correct session and the paper name
        Pre-process the session names to only keep the session name rather than including keywork: "SESSION:"
        """
        df_2 = df.replace(np.nan, "")
        session_data = df_2.index[[df_2["sessions-and-papers"].str.contains("SESSION:")]].values  # get all sessions
        session_data = list(session_data)
        current_index = 0

        while session_data:  # Iterate sessions to create papers within the session in order

            # Get index of session and session name
            current_index = session_data.pop(0)
            session_name = df_2.iloc[current_index]["sessions-and-papers"]
            session_name = session_name.split("SESSION: ")[1]

            # Get paper names
            paper_name = "lets get started"
            while "SESSION:" not in paper_name and paper_name != "":
                current_index = current_index + 1
                paper_name = df_2.iloc[current_index]["sessions-and-papers"]
                # print(paper_name)
                # print(type(paper_name))

                if paper_name not in self.papers:  # add the paper if it doesn't exist!
                    self.papers.update({paper_name: Paper(paper_name, session_name)})

                paper_name = df_2.iloc[current_index + 1]["sessions-and-papers"]

        print("DONE ITERATING FOR SESSIONS AND PAPER NAMES")
        return current_index + 1

    def extract_abstract(self, paper_name, data, data_index):
        """Check the pandas dataset for the abstract"""
        print(paper_name)
        abstract = data.iloc[data_index]["article-abstract"]
        self.papers.get(paper_name).add_abstract(abstract)

    def extract_value(self, paper_name, data, data_index, column_name, variable_name):
        """Depending on column name and variable name:
         We get data (value_name) from 'column_name' from the pandas dataset, and
         add the value to the particular key ('variable_name') within the paper's dictionary.

         Keyword arguments:
         paper_name -- key for the papers dictionary
         data -- particular pandas dataset
         data_index -- given the peculiarity of the webscraper, each row contain a different datapoint for the paper
         column_name -- What column should we take from the pandas dataset ('data')
         variable_name -- key in the dictionary within particular Paper object where we will add a new value!
         """
        df_2 = data.replace(np.nan, "")  # Replace all NaN values with empty strings (for easier iteration)
        value_name = "lets get started"
        current_sections = 0
        while value_name != "":  # we iterate the
            data_index = data_index + 1
            value_name = df_2.iloc[data_index][column_name]
            # print(paper_name)
            # print(type(paper_name))

            if value_name == "":
                break

            if paper_name in self.papers:
                self.papers[paper_name].add_value(value_name, variable_name)

            if variable_name == "section":
                current_sections = current_sections + 1

        if current_sections > self.max_sections:
            self.max_sections = current_sections
            self.max_sections_paper_name = paper_name

        return data_index - 1

    def get_pdf_files(self, directory, extension : str):
        filenames = listdir(directory)
        files = [filename for filename in filenames if filename.endswith(extension) and not filename.startswith("~")]

        return files

    def starts_with_figure_number(self, text):
        # Define the regex pattern
        patterns = [r"^Figure \d+", r"^Fig. \d+", r"^fig. \d+"]

        combined_pattern = '|'.join(patterns)

        # Search for the pattern in the text
        match = re.match(combined_pattern, text)

        # Return True if a match is found, else False
        return bool(match)

    def starts_with_table_number(self, text):
        # Define the regex pattern
        patterns = [r"^Table \d+", r"^Tab. \d+", r"^tab. \d+"]

        combined_pattern = '|'.join(patterns)

        # Search for the pattern in the text
        match = re.match(combined_pattern, text)

        # Return True if a match is found, else False
        return bool(match)

    def combine_positions(self, table_positions, min_threshold=30):
        combined_positions = {"x0": [], "x1": [], "y0": [], "y1": []}

        if not table_positions["x0"]:
            return table_positions

        # Initial positions
        current_x0 = table_positions["x0"][0]
        current_x1 = table_positions["x1"][0]
        current_y0 = table_positions["y0"][0]
        current_y1 = table_positions["y1"][0]

        for i in range(1, len(table_positions["x0"])):
            next_x0 = table_positions["x0"][i]
            next_x1 = table_positions["x1"][i]
            next_y0 = table_positions["y0"][i]
            next_y1 = table_positions["y1"][i]

            # Check if the difference between current y1 and next y0 is within the threshold
            if min_threshold <= next_y0 - current_y1:
                # Save the current combined positions
                combined_positions["x0"].append(current_x0)
                combined_positions["x1"].append(current_x1)
                combined_positions["y0"].append(current_y0)
                combined_positions["y1"].append(current_y1)

                # Move to the next position
                current_x0 = next_x0
                current_x1 = next_x1
                current_y0 = next_y0
                current_y1 = next_y1
            else:
                # Combine the positions
                current_x1 = max(current_x1, next_x1)
                current_y1 = next_y1

        # Append the last set of combined positions
        combined_positions["x0"].append(current_x0)
        combined_positions["x1"].append(current_x1)
        combined_positions["y0"].append(current_y0)
        combined_positions["y1"].append(current_y1)

        return combined_positions

    def check_if_header_footer(self, bounds, page_bounds, page_bound_perc=0.1):
        if 50 < bounds[1] < page_bounds[3] - (page_bounds[3] * page_bound_perc):
            return False
        else:
            return True

    def extract_section_names(self, file_path):
        pdf_document = pymupdf.open(file_path)
        # print(pdf_document.get_toc())

        document_toc = pdf_document.get_toc()

        # Extracting the second element where the first element is 1 (1 means that it is a section!)
        extracted_sections = [sublist[1] for sublist in document_toc if sublist[0] == 1]
        # print(extracted_sections)
        # print(pdf_document.chapter_count)
        #print(pdf_document.extract_font(4711))
        # print(pdf_document.get_page_fonts(0, full=True))
        return extracted_sections

    def extract_font_sizes(self, file_path):
        Extract_Data = []
        Font_size = 0.0
        for page_layout in extract_pages(file_path):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for text_line in element:
                        # print(text_line)
                        if text_line.get_text() != "" and not isinstance(text_line, LTChar) and not isinstance(text_line, LTAnno):
                            for character in text_line:
                                if isinstance(character, LTChar):
                                    Extract_Data.append([round(character.size, 2), text_line.get_text()])
                                break
 #                   Extract_Data.append([Font_size, (element.get_text())])

        # print(Extract_Data)
        return Extract_Data

    def check_if_text_in_block(self, text, testing_index):
        block_text = text[testing_index]
        if not testing_index:  # This means that the text is one of those without abstract title
            abstract_index = self.find_abstract_alternative(testing_index)

            if abstract_index:
                abstract = testing_index[abstract_index[0][0] - 1]  # this is actually CCS Concepts
                abstract_exist = True
        else:
            abstract = testing_index[0][1]
            abstract_exist = True

        if abstract_exist:
            abstract = abstract.replace("\n", " ")
            abstract = abstract.strip()
            self.papers[testing_index].add_abstract(abstract)

    def extract_text_from_pdf(self, file_path):

        extra_margin_tables = 10

        # Open the PDF file
        pdf_document = pymupdf.open(file_path)

        # print(pdf_document.page_count)
        # print(pdf_document.metadata["author"])
        # print(pdf_document.metadata["keywords"])
        # print(pdf_document.metadata["title"])
        #
        # print(pdf_document.get_toc())

        extracted_text = []

        html_page = pdf_document[0].get_text("html")
        xml_page = pdf_document[0].get_text("xml")

        for page_num in range(len(pdf_document)):
            # print("PAGE NUMBER::: " + str(page_num))
            page = pdf_document[page_num]
            text = page.get_text("blocks")
            #text_0 = page.get_text("text")
            #text_1 = page.get_text("words")
            table_positions = {"x0": [], "x1": [], "y0": [], "y1": []}
            page_bounds = page.bound()

            tabs = page.find_tables()
            # print(f"{len(tabs.tables)} table(s) on {page}")

            for tab in tabs:
                table_positions["x0"].append(tab.bbox[0] - extra_margin_tables)
                table_positions["x1"].append(tab.bbox[2] + extra_margin_tables)
                table_positions["y0"].append(tab.bbox[1] - extra_margin_tables)
                table_positions["y1"].append(tab.bbox[3] + extra_margin_tables)

                # for line in tab.extract():
                #     print(line)

            table_positions = self.combine_positions(table_positions)
            block_index = -1
            abstract_found = False

            for block in text:
                block_index += 1
                # block is a tuple: (x0, y0, x1, y1, "text", block_no, block_type)
                add_text = True
                for table_numbers in range(len(table_positions['x0'])):
                    # The text is within the height bounds of the table (so... possible text within a table)
                    if block[1] > table_positions['y0'][table_numbers] and block[1] < table_positions['y1'][
                        table_numbers]:

                        # Now we check if the text is within the width bounds of the table
                        if block[0] < table_positions['x1'][table_numbers]:

                            # FIXME: Hack in case the abstract is a table (sometimes)... Shuold be fix to add all the abstract, but it is good for now!
                            if 'abstract' in block[4] or 'Abstract' in block[4] or "ABSTRACT" in block[4] or "abstract." in block[4] or "Abstract." in block[4] or "ABSTRACT." in block[4]:
                                if len(block[4]) < 20: # hack in case the block is just "abstract"
                                    abstract_found = True

                                add_text = True
                                break
                            elif abstract_found:
                                abstract_found = False
                                add_text = True
                                break

                            add_text = False
                            break

                if not add_text:
                    continue
                elif self.starts_with_figure_number(block[4]) or self.starts_with_table_number(block[4]):
                    continue
                elif ((block[4].startswith("Permission to make") or block[4].startswith("ACM Reference Format")
                      or block[4].startswith("Authors’ addresses:") or block[4].startswith(
                            "Publication rights licensed")) or block[4].startswith("Proceedings of the ") or
                      block[4] == " \n") or block[4].startswith("Copyright"):
                    continue
                elif self.check_if_header_footer(block, page_bounds):
                    continue

                # text_content = block[4].encode('utf-8').decode('utf-8')

                text_content = self.replace_special_characters_followed_by_newline(block[4])
                extracted_text.append(text_content)
                # print(block[4])
                # if block[6] != 0:
                #     print(block[6])
        pdf_document.close()
        return extracted_text

    def get_sections_from_font_size(self, font_lists):
        sections = []
        bigger_size_headers = []
        different_sections = False
        reference_font_size = 0.0

        # Define the regular expression pattern
        # TODO: Make it some references is uncased.
        pattern = re.compile(r'^\s*\d*\W*\s*[^\S\n]*REFERENCES[^\S\n]*.*$')

        matches = [(font_size, text) for font_size, text in font_lists if pattern.match(text)]
        # pattern.search #?

        if matches: # todo: Problem is here with the references.... think about what to do?
            # print(matches)
            reference_font_size = matches[0][0]
        else:
            return None


        # for font_size, text in font_lists:
        #     if text.startswith("References") or text.startswith("references") or text.startswith("REFERENCES"):
        #         reference_font_size = font_size
        #         break

        for font_size, text in font_lists:
            if reference_font_size - 0.2 < font_size < reference_font_size + 0.2:
                sections.append(text)
            elif font_size > reference_font_size + 0.2:
                bigger_size_headers.append((font_size, text))

        # This two parts of the code should only happen if the REferences header is smaller (only a few cases!)
        for bigger_size_font, bigger_size_header in bigger_size_headers:
            if (("Introduction" in bigger_size_header) or ("introduction" in bigger_size_header) or ("INTRODUCTION" in bigger_size_header) or
                    ("Conclusions" in bigger_size_header) or ("conclusions" in bigger_size_header) or ("CONCLUSIONS" in bigger_size_header)):
                different_sections = True
                reference_font_size = bigger_size_font
                break

        if different_sections:
            sections.clear()
            sections.append(matches[0][1])
            for font_size, text in font_lists:
                if reference_font_size - 0.2 < font_size < reference_font_size + 0.2:
                    sections.append(text)

        return sections

    def extract_references(self, reference_text):
        reference_list = []
        current_reference = ""
        possible_end = False
        pattern = re.compile(r'^[A-Z][a-z]*,')
        pattern_2 = re.compile(r'^[A-Z][a-z]*.,')
        start_pattern = re.compile(r'(^\[\d+\])|(\[\d+\]$)')
        first = True

        pattern1 = re.compile(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*,.*?\d{4}\..*?(?=(?:[A-Z][a-z]+,)|$))', re.DOTALL)
        pattern1 = re.compile(
            r'([A-Z][a-zA-Z.,\-\s]+?\d{4}\..*?(?:[A-Z]{2,}.*?|\d{1,4}–\d{1,4}\.|$))',
            re.DOTALL
        )
        pattern1 = re.compile(
            r'([A-Z][a-zA-Z.,\-\s]+?\d{4}\..*?(?:\bIn\b.*?\.|IEEE Trans\..*?\.|Proc\..*?\.|$))',
            re.DOTALL
        )
        pattern1 = re.compile(
            r'([A-Z][a-zA-Z.,\-\s]+?\d{4}\..*?(?:\.\s|$))',
            re.DOTALL
        )
        matches = pattern1.findall(reference_text)

        # Separate regex patterns for traditional references and online references
        pattern_traditional = re.compile(
            r'([A-Z][a-zA-Z.,\-\s]+?\d{4}.*?(?:\d{1,4}–\d{1,4}\.|(?:In Proc|IEEE|ACM|AAAI|GECCO|AIIDE).*?\.\s|$))',
            re.DOTALL
        )
        pattern_online = re.compile(
            r'([A-Z][a-zA-Z.,\-\s]+?Online; accessed.*?https?:\/\/\S+\.?)',
            re.DOTALL
        )

        # Apply both patterns to capture traditional and online references
        traditional_references = pattern_traditional.findall(reference_text)
        online_references = pattern_online.findall(reference_text)

        # Combine the results
        strict_cleaned_references = traditional_references + online_references

        # Output the extracted references
        # for i, match in enumerate(all_references_combined, 1):
        #     print(f"Reference {i}:\n{match}\n")

        pattern_strict_references = re.compile(
            r'(?:\[\d+\]\s[A-Z].*?(?=\[\d+\]|\Z))|(?:[A-Z][a-zA-Z.,\-\s]+?,\s\d{4}.*?(?:\d{1,4}–\d{1,4}\.|https?:\/\/\S+|\Z))',
            re.DOTALL
        )

        pattern_strict_references = pattern_extended = re.compile(
            r'(\[\d+\]\s(?:\d{4}\.|[A-Z][a-zA-Z.,\-\s]+?)\s.*?(?=\[\d+\]|\Z))',
            re.DOTALL
        )

        pattern_strict_references= re.compile(
            r'(?:\[\d+\]\s(?:\d{4}\.|[A-Z][a-zA-Z.,\-\s]+?)\s.*?(?=\[\d+\]|\Z))'  # Matches [xxx] references (including year-first ones)
            r'|(?:[A-Z][a-zA-Z.,\-\s]+?,\s\d{4}.*?(?:\d{1,4}–\d{1,4}\.|https?:\/\/\S+|\Z))',  # Matches traditional references
            re.DOTALL
        )

        pattern_strict_references = re.compile(
            r'(?:\[\d+\]\s*\n*(?:\d{4}\.|[A-Z][a-zA-Z.,\-\s]+?)\s.*?(?=\[\d+\]|\Z))'  # Matches [xxx] references (allowing optional newlines)
            r'|(?:[A-Z][a-zA-Z.,\-\s]+?,\s\d{4}.*?(?:\d{1,4}–\d{1,4}\.|https?:\/\/\S+|\Z))',  # Matches traditional references
            re.DOTALL
        )


        # pattern_strict_references = re.compile(
        #     r'(?:\[\d+\]\s(?:[n.d.]\.|[A-Z][a-zA-Z.,\-\s]+?)\s.*?(?=\[\d+\]|\Z))'
        #     r'(?:\[\d+\]\s(?:\d{4}\.|[A-Z][a-zA-Z.,\-\s]+?)\s.*?(?=\[\d+\]|\Z))'  # Matches [xxx] references (including [n. d.])
        #     r'|(?:[A-Z][a-zA-Z.,\-\s]+?,\s\d{4}.*?(?:\d{1,4}–\d{1,4}\.|https?:\/\/\S+|\Z))',  # Matches traditional references
        #     re.DOTALL
        # )

        # Apply the strict regex to capture all valid references
        # TODO: Use this for general extraction. Remove for specific use case (raluca paper)
        strict_cleaned_references = pattern_strict_references.findall(reference_text)

        # for i, match in enumerate(strict_cleaned_references, 1):
        #     print(f"Reference {i}:\n{match}\n")

        return strict_cleaned_references

        if not reference_text.startswith("["):
            reference_list = all_references_combined
        else:
            for line in reference_text.split('\n'):
                # print(line)
                stripped_line = line.strip()
                if (possible_end and (bool(pattern.match(stripped_line)) or bool(pattern_2.match(stripped_line)))) or bool(start_pattern.match(stripped_line)) and not first:
                    reference_list.append(current_reference)
                    current_reference = ""

                possible_end = False

                if stripped_line.endswith("."):
                    possible_end = True

                current_reference += stripped_line + " "
                first = False

            if current_reference != "":
                reference_list.append(current_reference)


        return reference_list

    def replace_special_characters_followed_by_newline(self, text_content):
        # Define the regular expression pattern
        pattern = re.compile(r'[~¨˜]\n([a-zA-Z])')

        # Define the replacement function
        def replacement(match):
            return match.group(1)

        # Replace the pattern in each string
        result = pattern.sub(replacement, text_content)
        return result

    def load_all_papers_pdf(self, papers_directory):
        folders = listdir(papers_directory)
        if '.DS_Store' in folders:
            folders.remove('.DS_Store')

        print(folders)

        # now we iterate by year basically
        for folder in folders:
            proceedings_year = folder
            print(proceedings_year)
            folder_in_year = str(papers_directory) + "/" + folder + "/"
            current_folder = folder_in_year
            files_in_folder = listdir(current_folder)
            if '.DS_Store' in current_folder:
                current_folder.remove('.DS_Store')

            # now iterate the papers per year
            files = [filename for filename in files_in_folder if
                     filename.endswith(".pdf") and not filename.startswith("~")]

            for file in files:
                paper_directory = current_folder + "/" + file
                # section_names = self.extract_section_names(paper_directory)
                section_names = self.extract_font_sizes(paper_directory)
                section_names = self.get_sections_from_font_size(section_names)

                if section_names is None: ## TODO: This will stop for now papers without References!
                    print("References Problems with: " + file + ", year: " + str(proceedings_year) + ", type: " + str(
                        "main"))
                    continue

                text_data = self.extract_text_from_pdf(paper_directory)
                text_data = self.remove_random_header(text_data)

                if text_data[0][0].isdigit():
                    del text_data[0]

                # text_data now contains everything from the paper (everything interesting!)
                # Now, we need to extract the info needed for the Paper file

                # first the title
                paper_title = text_data[0]
                cleaned_paper_title = paper_title.replace("\n", " ")
                cleaned_paper_title = cleaned_paper_title.strip()
                print(cleaned_paper_title)
                self.papers.update({cleaned_paper_title: Paper(cleaned_paper_title, proceedings_year, "main", file)})

                # now the bendito abstract
                abstract_index = self.find_abstract(text_data)
                abstract_exist = False

                if not abstract_index: # This means that the text is one of those without abstract title
                    abstract_index = self.find_abstract_alternative(text_data)

                    if abstract_index:
                        abstract = abstract_index[0][1]
                        abstract_exist = True
                else:
                    abstract = abstract_index[0][1]
                    abstract_exist = True

                if abstract_exist:
                    abstract = abstract.replace("\n", " ")
                    abstract = abstract.strip()
                    self.papers[cleaned_paper_title].add_abstract(abstract)

                    # and now the big part, extract the sections!
                    current_text_data = text_data[abstract_index[0][0] + 1:]
                else:
                    self.papers[cleaned_paper_title].add_abstract("there was no abstract!")
                    current_text_data = text_data

                # possible_sections = self.find_sections(current_text_data)
                possible_sections = self.find_sections_with_names(current_text_data, section_names)
                #possible_sections, current_text_data = self.find_sections_with_names_data_extraction(current_text_data, section_names)
                possible_sections.sort(key=lambda x: x[0])

                # print(possible_sections)
                if not possible_sections:
                    print("Section Problems with: " + file + ", year: " + str(proceedings_year) + ", type: " + str("main"))
                else:
                    if possible_sections[0][0] != 0: # FIXME: Not sure about this change!
                        possible_sections.insert(0, (0, current_text_data[0]))

                iteration = 0

                for possible_section in possible_sections:
                    iteration += 1
                    section_index = possible_section[0]
                    section_text = possible_section[1]
                    final_section_text = ""
                    acm_keywords = False
                    keywords = False

                    keyword_text = section_text
                    keyword_text = keyword_text.strip()
                    # keyword_text = keyword_text.split()

                    if (keyword_text.startswith("CCS CONCEPTS") or keyword_text.startswith("CCS Concepts") or keyword_text.startswith("ccs concepts")
                            or keyword_text.startswith("Categories") or keyword_text.startswith("General Terms")):
                        acm_keywords = True
                    elif keyword_text.startswith("KEYWORDS") or keyword_text.startswith("keywords") or keyword_text.startswith("Keywords") or keyword_text.startswith("keyword"):
                        keywords = True

                    # we need to do work here to extract the section title!
                    # section_title = self.extract_until_newline(section_text)
                    # section_title = section_title.strip()

                    reference_text = section_text
                    reference_text = reference_text.strip()
                    reference_text = reference_text.split()
                    reference_text_max_length = len(reference_text)

                    if reference_text_max_length > 2:
                        reference_text_max_length = 2

                    if reference_text[0].startswith("References") or reference_text[0].startswith("references") or reference_text[0].startswith("REFERENCES"):
                        reference_text = reference_text[0]
                    else:
                        reference_text = reference_text[reference_text_max_length - 1]




                    if reference_text.startswith("References") or reference_text.startswith("references") or reference_text.startswith("REFERENCES"):
                        # print("WE ARE DONE WITH THIS!")
                        current_text_data[section_index].replace("REFERENCES\n", "")
                        current_text_data[section_index].replace("references\n", "")
                        current_text_data[section_index].replace("References\n", "")
                        for step in range(section_index, len(current_text_data)):
                            current_section_text = current_text_data[step]
                            current_section_text = current_section_text.replace("REFERENCES\n", "")
                            current_section_text = current_section_text.replace("references\n", "")
                            current_section_text = current_section_text.replace("References\n", "")

                            current_section_text = current_section_text.replace("REFERENCES \n", "")
                            current_section_text = current_section_text.replace("references \n", "")
                            current_section_text = current_section_text.replace("References \n", "")

                            reference_list = self.extract_references(current_section_text)

                            for reference in reference_list:
                                reference = reference.replace("-\n", "")
                                reference = reference.replace("\n", " ")
                                self.papers[cleaned_paper_title].add_value(reference, "reference")

                    else:
                        if iteration > len(possible_sections) - 1:
                            next_section_index = len(current_text_data)
                        else:
                            next_section_index = possible_sections[iteration][0]

                        for step in range(section_index, next_section_index):
                            current_section_text = current_text_data[step]
                            current_section_text = current_section_text.replace("-\n", "")
                            current_section_text = current_section_text.replace("\n", " ")
                            final_section_text += current_section_text

                        if acm_keywords:
                            self.papers[cleaned_paper_title].add_value(final_section_text, "acm_keywords")
                        elif keywords:
                            self.papers[cleaned_paper_title].add_value(final_section_text, "keywords")
                        else:
                            self.papers[cleaned_paper_title].add_value(final_section_text, "section")

        self.pdfs_to_compressed_view()

    def remove_illegal_characters(self, value):
        if isinstance(value, str):
            return ''.join(
                char for char in value if char.isprintable() and (ord(char) > 31 or char in ('\t', '\n', '\r')))
        return value

    def pdfs_to_compressed_view(self):

        # Get the paper with the most sections!
        self.collected_papers_general = CollectedPapersGeneral("aiide-dataset", self.max_sections)

        for paper_name, paper_object in self.papers.items():
            print(paper_name)

            self.collected_papers_general.add_sections(paper_object.get_paper_sections_raw())

            self.collected_papers_general.add_full_datapoint(
                paper_object.get_paper_name(),
                paper_object.get_paper_session(),
                paper_object.get_paper_proceedings(),
                paper_object.get_paper_abstract(),
                paper_object.get_paper_authors(),
                paper_object.get_paper_full_authors(),
                paper_object.get_paper_section_titles(),
                # paper_object.get_paper_sections(),
                paper_object.get_paper_references(),
                paper_object.get_paper_acm_keywords(),
                paper_object.get_paper_keywords()
            )

        df = pd.DataFrame(self.collected_papers_general.get_data(), columns=self.collected_papers_general.get_data().keys())
        df_cleaned = df.applymap(self.remove_illegal_characters)
        #df.to_excel("output.xlsx")
        # df_cleaned.to_excel(self.papers_directory + "/../aiide-dataset-peryeartest-third-nocontains.xlsx")
        # df_cleaned.to_excel(self.papers_directory + "/../aiide-dataset-fulltest.xlsx")
        # df_cleaned.to_excel(self.papers_directory + "/../FDG-dataset-minitest.xlsx")
        # df_cleaned.to_excel(self.papers_directory + "/../fdg-error-dataset.xlsx")
        df_cleaned.to_excel(self.papers_directory + "/../FDG-dataset.xlsx")


    def extract_until_newline(self, string):
        pattern = re.compile(r".*?(?=\n)")
        # pattern = re.compile(r"\D*?(?=\n)")
        result = pattern.search(string)
        if result:
            return result.group()
        else:
            return ""
    def find_sections_with_names(self, text_data, known_sections):
        matches = []

        for known_section in known_sections:
            known_section_strip = known_section.replace("\n", " ")
            known_section_strip = " ".join(known_section.split())
            for index, text in enumerate(text_data):
                aux_text = text.replace("\n", " ")
                aux_text = " ".join(text.split())
                if aux_text.startswith(known_section_strip):
                    matches.append((index, text))
                    break
                # elif ## todo: What I am trying to do here is when the text includes the section name (but it does not start with it!...)

        return matches

    def find_sections_with_names_data_extraction(self, text_data, known_sections):
        matches = []
        aux_text_data = text_data

        for known_section in known_sections:
            known_section_strip = known_section.replace("\n", " ")
            known_section_strip = " ".join(known_section.split())
            for index, text in enumerate(aux_text_data):
                aux_text = text.replace("\n", " ")
                aux_text = " ".join(text.split())
                if aux_text.startswith(known_section_strip):
                    matches.append((index, text))
                    break
                # elif ## todo: What I am trying to do here is when the text includes the section name (but it does not start with it!...)
                elif known_section_strip in aux_text:
                    within_index = aux_text.find(known_section_strip)
                    del aux_text_data[index]
                    aux_text_data.insert(index - 1, aux_text[:within_index])
                    aux_text_data.insert(index, aux_text[within_index:])
                    matches.append((index, aux_text[within_index:]))
                    break

        return matches, aux_text_data

    def find_sections(self, text_data):

        patterns = [r"^\b[A-Z]+ \n", r"^\b[A-Z]+\n",
                    r"\b\d+ [A-Z]+\n", r"\b\d+ [A-Z]+ \n",
                    r"\b\d+\n+[A-Z]+\n", r"\b\d+\n+[A-Z]+ \n",
                    r"^\d+\n(?:[A-Z ]+)+\n", r"^\d+\n(?:[A-Z ]+)+ \n",
                    r"^\d+\n(?:[A-Z &]+\n)", r"^\d+\n(?:[A-Z &]+ \n)",
                    r"^\d+\n(?:[A-Z :\-.&]+\n)", r"^\d+\n(?:[A-Z :\-.&]+ \n)"
                    ]
        combined_pattern = '|'.join(patterns)

        # matches = [value for value in text_data if re.match(combined_pattern, value)]
        matches = [(index, value) for index, value in enumerate(text_data) if  re.match(combined_pattern, value)]

        return matches

    def remove_random_header(self, text_data):

        #patterns = [r"\d+:\d+\n"]
        #combined_pattern = '|'.join(patterns)

        pattern = re.compile(r"\d+:\d+\n")
        filtered_strings = [value for value in text_data if not pattern.search(value)]
        # filtered_strings = [value for value in text_data if not re.match(combined_pattern, value)]

        return filtered_strings

    def find_abstract(self, text_data):
        next_index = False
        index_to_continue = 0
        pattern = re.compile(r"^\s*abstract", re.IGNORECASE)
        matches = [(index, string) for index, string in enumerate(text_data) if pattern.match(string)]
        for match in matches:
            aux_text = match[1].replace("\n", "")
            aux_text = aux_text.strip()
            if aux_text == 'abstract' or aux_text == 'Abstract' or aux_text == "ABSTRACT" or aux_text == "abstract." or aux_text == "Abstract." or aux_text == "ABSTRACT.":
                next_index = True
                index_to_continue = match[0] + 1
                break

        if next_index:
            matches = [(index_to_continue, text_data[index_to_continue])]

        return matches

    def find_abstract_alternative(self, text_data):
        # pattern = re.compile(r"^\s*CCS Concepts", re.IGNORECASE)
        # matches = [(index, string) for index, string in enumerate(text_data) if pattern.match(string)]
        next_index = False
        index_to_continue = 0
        abstract_strings = [(index, s) for index, s in enumerate(text_data) if 'abstract' in s.lower()]
        match = []
        abstract = ""
        if abstract_strings:
            abstract = abstract_strings[0][1][abstract_strings[0][1].find('Abstract'):]
            aux_text = abstract.replace("\n", "")
            aux_text = aux_text.strip()

            if aux_text == 'abstract' or aux_text == 'Abstract' or aux_text == "ABSTRACT":
                next_index = True
                index_to_continue = abstract_strings[0][0] + 1
            else:
                index_to_continue = abstract_strings[0][0]

        if next_index:
            match = [(index_to_continue, text_data[index_to_continue])]
        else:
            match = [(index_to_continue, abstract)]

        return match

    def load_all_original_papers_data(self):
        xlsx_files = self.get_files(self.papers_directory, ".xlsx")  # Get all excel files (for now CHI23)
        current_index = 0
        current_paper_number = 0
        max_papers_assessed = 20

        for file in xlsx_files:
            print(file)

            current_data = pd.read_excel(self.papers_directory + file)  # Get the pandas object

            current_index = 1 # We start from 1 because 0 is always -1 for the validation data we use
            current_index = self.extract_sessions(current_data)  # Extract the session data and pre-process
            print(current_index)
            print(current_data.iloc[[current_index]])
            print(current_data.iloc[[0]])
            print(len(current_data))

            while current_index + 1 < len(current_data):
                # current_paper_number = current_paper_number + 1
                # if current_paper_number > max_papers_assessed:
                #     break

                current_paper_name = current_data.iloc[current_index]["article-open"]
                # current_paper_name = current_paper_name.replace(" :", ":")
                self.extract_abstract(current_paper_name, current_data, current_index)

                variable_index = 0
                current_index = current_index - 1 # little hack for sergberto
                for column_name in self.column_names:
                    current_index = self.extract_value(current_paper_name, current_data, current_index, column_name, self.variable_names[variable_index])
                    variable_index = variable_index + 1

                next_paper_name = current_paper_name
                while current_paper_name == next_paper_name and current_index + 1 < len(current_data):
                    current_index = current_index + 1
                    next_paper_name = current_data.iloc[current_index]["article-open"]
                    # next_paper_name = current_data.iloc[current_index]["article-titles"]

            print("are we done?")  # Well not really, now I need to make it into an excel.
            print(self.max_sections)
            self.papers_to_compressed_view()

    def load_compressed_view(self):
        xlsx_files = self.get_files(self.papers_directory, ".xlsx")  # Get the compressed view
        self.papers = {}

        for file in xlsx_files:
            print(file)

            current_data = pd.read_excel(self.papers_directory + "/" + file, engine='openpyxl')  # Get the pandas object
            current_data = current_data.replace(np.nan, "")
            current_data = current_data.replace("&nbsp", "")
            current_data = current_data.replace('\u00A0', '')

            for index, row in current_data.iterrows():

                paper_name = row["paper_name"]
                paper_session = row["paper_session"]
                paper_year = row["paper_proceeding"]
                paper = Paper(paper_name, paper_year, paper_session, "")

                print(paper_name)

                if paper_name not in self.papers:  # add the paper if it doesn't exist!
                    self.papers.update({paper_name: paper})

                for (columnName, columnData) in current_data.iteritems():

                    if columnName == 'paper_name' or columnName == 'paper_session':
                        continue
                    else:
                        paper.add_value_compressed(row[columnName], columnName)

        print("done!")

    def calculate_papers_metrics(self, column_name: str, avg = True, sd = True, file_name="paper_metrics.csv", save=True, append=True):

        total_count = 0
        paper_count = 0
        counter = []
        avg_count = 0
        sd_count = 0

        for paper_name, paper in self.papers.items():
            paper_count += 1
            total_count += len(paper.get_paper_full_authors_no_delimiter())
            counter.append(len(paper.get_paper_full_authors_no_delimiter()))

        counter = np.array(counter)
        avg_count = float(total_count)/float(paper_count)

        print(avg_count)
        print("Counting authors per paper \n \n")

        print(len(counter))
        print(np.average(counter))
        print(np.std(counter))
        print(np.percentile(counter, 2.5))
        print(np.percentile(counter, 97.5))

    def calculate_metrics_sections(self, column_name: str, avg = True, sd = True, file_name="paper_metrics.csv", save=True, append=True):

        total_count = 0
        paper_count = 0
        counter = []
        avg_count = 0
        sd_count = 0

        for paper_name, paper in self.papers.items():
            paper_count += 1
            total_count += len(paper.get_paper_section_titles_no_delimiter())
            counter.append(len(paper.get_paper_section_titles_no_delimiter()))

        counter = np.array(counter)

        print("Counting sections \n \n")

        print(len(counter))
        print(np.average(counter))
        print(np.std(counter))
        print(np.percentile(counter, 2.5))
        print(np.percentile(counter, 97.5))

    def calculate_metrics_references(self, column_name: str, avg = True, sd = True, file_name="paper_metrics.csv", save=True, append=True):

        total_count = 0
        paper_count = 0
        counter = []
        avg_count = 0
        sd_count = 0

        for paper_name, paper in self.papers.items():
            paper_count += 1
            total_count += len(paper.get_paper_references_no_delimiter())
            counter.append(len(paper.get_paper_references_no_delimiter()))

        counter = np.array(counter)

        print("Counting references \n \n")

        print(len(counter))
        print(np.average(counter))
        print(np.std(counter))
        print(np.percentile(counter, 2.5))
        print(np.percentile(counter, 97.5))

    def calculate_metrics_for_sessions(self, column_name: str, avg = True, sd = True, file_name="paper_metrics.csv", save=True, append=True):

        total_count = 0
        paper_count = 0
        counter = []
        avg_count = 0
        sd_count = 0
        session_count = {}

        for paper_name, paper in self.papers.items():
            paper_count += 1

            if paper.get_paper_session() not in session_count:
                session_count.update({paper.get_paper_session(): 1})
            else:
                session_count[paper.get_paper_session()] += 1

        for session, count in session_count.items():
            counter.append(count)

        counter = np.array(counter)
        print("Counting sessions \n \n")
        print(len(counter))
        print(np.average(counter))
        print(np.std(counter))
        print(np.percentile(counter, 2.5))
        print(np.percentile(counter, 97.5))

    def calculate_ngram_per_paper(self, column_name: str, all_ngrams_until=True, ngram=3, top_k=5, avg=True, sd=True, file_name="paper_metrics.csv", save=True, append=True):

        if all_ngrams_until:
            grams = [Counter() for i in range(ngram)]
        else:
            grams = [Counter()]

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            paper_sections_no_punctuation = [self.remove_punctuation(x) for x in paper_sections]

            seen_ngrams = set()

            for section in paper_sections_no_punctuation:
                words = section.split()

                if all_ngrams_until:
                    for gram_step in range(ngram):
                        ngrams = [tuple(words[i:i + gram_step + 1]) for i in range(len(words) - gram_step)]
                        ngram_counts = Counter(ngrams)

                        # for ngram, count in sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                        #   print(f"{ngram}: {count} occurrences of total)")

                        for ngram_collected in ngrams:
                            # Count only if the n-gram hasn't been seen in this text before
                            if ngram_collected not in seen_ngrams:
                                grams[gram_step][ngram_collected] += 1
                                # all_ngram_counts[ngram] += 1
                                seen_ngrams.add(ngram_collected)

                        # grams[gram_step].update(ngram_counts)

                        # Update the overall counts
                        #all_ngram_counts.update(ngram_counts)
                else:
                    # Generate and count 3-grams
                    ngrams = [tuple(words[i:i + ngram]) for i in range(len(words) - ngram - 1)]
                    ngram_counts = Counter(ngrams)

                    for ngram_collected in ngrams:
                        # Count only if the n-gram hasn't been seen in this text before
                        if ngram_collected not in seen_ngrams:
                            grams[0][ngram_collected] += 1
                            # all_ngram_counts[ngram] += 1
                            seen_ngrams.add(ngram_collected)

                    # Update the overall counts
                    # grams[0].update(ngram_counts)


        # Display the top 5 n-grams by counts sorted!
        print("Here we present all the specified ngram ocurrances per paper!")

        for all_ngram_counts in grams:
            total_ngrams = len(self.papers)

            # all_ngram_counts = [sentence for sentence in sentences if not self.contains_keyword(sentence, self.ignore_keywords_analysis)]

            # Filtered Counter using dictionary comprehension
            filtered_counter = Counter({
                key: value for key, value in all_ngram_counts.items()
                if not self.contains_keyword_ngram(key, self.ignore_keywords_analysis)
            })

            for ngram, count in sorted(filtered_counter.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                percentage = (count / total_ngrams) * 100
                print(f"{ngram}: {count} occurrences ({percentage:.2f}% of total)")
            print("\n")

    # Function to check if any keyword is in a string
    def contains_keyword_ngram(self, tuple_key, keywords):
        words = ''.join(tuple_key)
        return any(keyword.lower() in words.lower() for keyword in keywords)
    def calculate_ngram(self, column_name: str, all_ngrams_until=True, ngram=3, top_k=5, avg=True, sd=True, file_name="paper_metrics.csv", save=True, append=True):

        if all_ngrams_until:
            grams = [Counter() for i in range(ngram)]
        else:
            grams = [Counter()]

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            paper_sections_no_punctuation = [self.remove_punctuation(x) for x in paper_sections]

            for section in paper_sections_no_punctuation:
                words = section.split()

                if all_ngrams_until:
                    for gram_step in range(ngram):
                        ngrams = [tuple(words[i:i + gram_step + 1]) for i in range(len(words) - gram_step)]
                        ngram_counts = Counter(ngrams)
                        grams[gram_step].update(ngram_counts)

                        # Update the overall counts
                        #all_ngram_counts.update(ngram_counts)
                else:
                    # Generate and count 3-grams
                    ngrams = [tuple(words[i:i + ngram]) for i in range(len(words) - ngram - 1)]
                    ngram_counts = Counter(ngrams)

                    # Update the overall counts
                    grams[0].update(ngram_counts)


        # Display the top 5 n-grams by counts sorted!
        print("Here we present all the specified ngram ocurrances in general in all papers and sections")
        for all_ngram_counts in grams:
            total_ngrams = sum(all_ngram_counts.values())
            for ngram, count in sorted(all_ngram_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                percentage = (count / total_ngrams) * 100
                print(f"{ngram}: {count} occurrences ({percentage:.2f}% of total)")
            print("\n")

    def extract_sentences(self, column_name: str, all_ngrams_until=True, minimum_length=1, top_k=5, avg=True, sd=True, file_name="paper_metrics.csv", save=True, append=True):

        all_sentence_counts = Counter()

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            paper_sections_no_punctuation = [self.remove_punctuation(x) for x in paper_sections]

            for section in paper_sections:

                # Preprocess the text (remove non-alphanumeric characters, convert to lowercase)
                text = re.sub(r'[^a-zA-Z.,;?!:\s\n\[\]\d-]', '', section)

                # Remove everything inside square brackets
                text = re.sub(r'\[.*?\]', '', text)

                text = re.sub(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]', '', text)

                # Remove punctuation even if there is a space in between, excluding "-"
                text = re.sub(r'([.,;])\s*(?<!\n)', r'\1', text)

                # Replace consecutive spaces with a single space, excluding "\n"
                # text = re.sub(r'(?<!\n)\s+', ' ', text)
                # text = re.sub(r'(?<!\n)\s+(?!\n)', ' ', text)

                # Remove spaces before punctuation, excluding "-"
                # text = re.sub(r'\s*([.,;?!:])\s*', r'\1', text)
                # text = re.sub(r'\s*([.,;?!:])', r'\1', text)
                text = re.sub(r'\s*([.,;?!:])(?<!\n)', r'\1', text)

                text = text.lower()

                # Extract sentences based on ".", ",", ";", and "\n"
                # sentences = re.split(r'[.,;\n]', text)

                # Extract sentences based on ".", ",", ";", ":", "!", "?", and "\n"
                sentences = re.split(r'[.,;?!:\n]', text)

                # Remove leading and trailing whitespaces from each sentence
                sentences = [sentence.strip() for sentence in sentences]

                # Filter out sentences that are either one word or empty space
                sentences = [sentence for sentence in sentences if len(sentence.split()) > minimum_length]

                # Count each unique sentence
                sentence_counts = Counter(sentences)

                # Update the overall counts
                all_sentence_counts.update(sentence_counts)


        # Display the top 5 n-grams by counts sorted!
        print(f"Here we present all the sentences ocurrances in general in all papers and sections with minimum_length {minimum_length}")
        total_sentences = sum(all_sentence_counts.values())
        for sentence, count in sorted(all_sentence_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            percentage = (count / total_sentences) * 100
            print(f"{sentence}: {count} occurrences ({percentage:.2f}% of total)")

        print("\n")

    def contains_keyword(self, sentence, keywords):
        return any(keyword.lower() in sentence.lower() for keyword in keywords)

    def extract_keywords_per_paper(self, column_name: str, all_ngrams_until=True, minimum_length=1, top_k=5, avg=True,
                                    sd=True, file_name="paper_metrics.csv", save=True, append=True):

        all_sentence_counts = Counter()

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            seen_sentence = set()

            for section in paper_sections:

                # Preprocess the text (remove non-alphanumeric characters, convert to lowercase)
                text = re.sub(r'[^a-zA-Z.,;?!:\s\n\[\]\d-]', '', section)

                # Remove everything inside square brackets
                text = re.sub(r'\[.*?\]', '', text)

                text = re.sub(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]', '', text)

                # Remove punctuation even if there is a space in between, excluding "-"
                text = re.sub(r'([.,;])\s*(?<!\n)', r'\1', text)

                # Replace consecutive spaces with a single space, excluding "\n"
                # text = re.sub(r'(?<!\n)\s+', ' ', text)
                # text = re.sub(r'(?<!\n)\s+(?!\n)', ' ', text)

                # Remove spaces before punctuation, excluding "-"
                # text = re.sub(r'\s*([.,;?!:])\s*', r'\1', text)
                # text = re.sub(r'\s*([.,;?!:])', r'\1', text)
                text = re.sub(r'\s*([.,;?!:])(?<!\n)', r'\1', text)

                text = text.lower()

                # Extract sentences based on ".", ",", ";", and "\n"
                # sentences = re.split(r'[.,;\n]', text)

                # Extract sentences based on ".", ",", ";", ":", "!", "?", and "\n"
                sentences = re.split(r'[.,;?!:\n]', text)

                # Remove leading and trailing whitespaces from each sentence
                sentences = [sentence.strip() for sentence in sentences]

                # Filter out sentences that are either one word or empty space
                sentences = [sentence for sentence in sentences if len(sentence.split()) > minimum_length]

                # Filter out based on keywords
                sentences = [sentence for sentence in sentences if
                             not self.contains_keyword(sentence, self.ignore_keywords_analysis)]

                # Count each unique sentence
                sentence_counts = Counter(sentences)

                for sentence_count in sentence_counts:
                    # Count only if the n-gram hasn't been seen in this text before
                    if sentence_count not in seen_sentence:
                        all_sentence_counts[sentence_count] += 1
                        # all_ngram_counts[ngram] += 1
                        seen_sentence.add(sentence_count)

        # Display the top 5 n-grams by counts sorted!
        print(f"Here we present all sentences ocurrances in papers with minimum_length {minimum_length}")
        total_sentences = len(self.papers)
        for sentence, count in sorted(all_sentence_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            percentage = (count / total_sentences) * 100
            print(f"{sentence}: {count} occurrences ({percentage:.2f}% of total)")

        print("\n")

    def extract_sentences_per_paper(self, column_name: str, all_ngrams_until=True, minimum_length=1, top_k=5, avg=True, sd=True, file_name="paper_metrics.csv", save=True, append=True):

        all_sentence_counts = Counter()

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            seen_sentence = set()

            for section in paper_sections:

                # Preprocess the text (remove non-alphanumeric characters, convert to lowercase)
                text = re.sub(r'[^a-zA-Z.,;?!:\s\n\[\]\d-]', '', section)

                # Remove everything inside square brackets
                text = re.sub(r'\[.*?\]', '', text)

                text = re.sub(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]', '', text)

                # Remove punctuation even if there is a space in between, excluding "-"
                text = re.sub(r'([.,;])\s*(?<!\n)', r'\1', text)

                # Replace consecutive spaces with a single space, excluding "\n"
                # text = re.sub(r'(?<!\n)\s+', ' ', text)
                # text = re.sub(r'(?<!\n)\s+(?!\n)', ' ', text)

                # Remove spaces before punctuation, excluding "-"
                # text = re.sub(r'\s*([.,;?!:])\s*', r'\1', text)
                # text = re.sub(r'\s*([.,;?!:])', r'\1', text)
                text = re.sub(r'\s*([.,;?!:])(?<!\n)', r'\1', text)

                text = text.lower()

                # Extract sentences based on ".", ",", ";", and "\n"
                # sentences = re.split(r'[.,;\n]', text)

                # Extract sentences based on ".", ",", ";", ":", "!", "?", and "\n"
                sentences = re.split(r'[.,;?!:\n]', text)

                # Remove leading and trailing whitespaces from each sentence
                sentences = [sentence.strip() for sentence in sentences]

                # Filter out sentences that are either one word or empty space
                sentences = [sentence for sentence in sentences if len(sentence.split()) > minimum_length]

                # Filter out based on keywords
                sentences = [sentence for sentence in sentences if not self.contains_keyword(sentence, self.ignore_keywords_analysis)]

                # Count each unique sentence
                sentence_counts = Counter(sentences)

                for sentence_count in sentence_counts:
                    # Count only if the n-gram hasn't been seen in this text before
                    if sentence_count not in seen_sentence:
                        all_sentence_counts[sentence_count] += 1
                        # all_ngram_counts[ngram] += 1
                        seen_sentence.add(sentence_count)


        # Display the top 5 n-grams by counts sorted!
        print(f"Here we present all sentences ocurrances in papers with minimum_length {minimum_length}")
        total_sentences = len(self.papers)
        for sentence, count in sorted(all_sentence_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            percentage = (count / total_sentences) * 100
            print(f"{sentence}: {count} occurrences ({percentage:.2f}% of total)")

        print("\n")

    def save_encountered_papers_keywords(self, searched_keyword):

        # Get the paper with the most sections!
        self.encountered_papers_general = EncounteredPapersGeneral("aiide-dataset", 50)

        for paper_name, paper_object in self.encountered_papers.items():
            print(paper_name)

            self.encountered_papers_general.add_sections(paper_object.get_paper_sections_raw())

            self.encountered_papers_general.add_full_datapoint(
                paper_object.get_paper_name(),
                paper_object.paper_number,
                paper_object.paper_counter,
                paper_object.get_paper_session(),
                paper_object.get_paper_proceedings(),
                paper_object.keyword
            )

        df = pd.DataFrame(self.encountered_papers_general.get_data(),
                          columns=self.encountered_papers_general.get_data().keys())
        df_cleaned = df.applymap(self.remove_illegal_characters)
        # df.to_excel("output.xlsx")
        # df_cleaned.to_excel(self.papers_directory + "/../aiide-dataset-peryeartest-third-nocontains.xlsx")
        df_cleaned.to_excel(self.papers_directory + "/../results_keywords/" + searched_keyword + ".xlsx")

    def extract_statistics(self, searched_keyword, dataset, print_total):

        # [total number, total non repetitive number, total main conference, total workshops]
        numbers_per_year = {}

        for i in range(2005, 2024):
            numbers_per_year.update({i: []})
            numbers_per_year[i].append(0)
            numbers_per_year[i].append(0)
            numbers_per_year[i].append(0)
            numbers_per_year[i].append(0)

        for paper_name, paper_object in dataset.items():

            paper_year = paper_object.get_paper_proceedings()

            if paper_year in numbers_per_year:
                # Then do something
                numbers_per_year[paper_year][0] += 1

                if paper_object.paper_counter == 1:
                    numbers_per_year[paper_year][1] += 1

                if paper_object.get_paper_session().startswith("ws"):
                    numbers_per_year[paper_year][3] += 1
                else:
                    numbers_per_year[paper_year][2] += 1

            else:
                numbers_per_year.update({paper_year : []})
                numbers_per_year[paper_year].append(0)
                numbers_per_year[paper_year].append(0)
                numbers_per_year[paper_year].append(0)
                numbers_per_year[paper_year].append(0)

                numbers_per_year[paper_year][0] += 1

                if paper_object.paper_counter == 1:
                    numbers_per_year[paper_year][1] += 1

                if paper_object.get_paper_session().startswith("ws"):
                    numbers_per_year[paper_year][3] += 1
                else:
                    numbers_per_year[paper_year][2] += 1

        print("-----------------------")

        print("NOW THE DATA PER YEAR!! " + searched_keyword)

        numbers_per_year = dict(sorted(numbers_per_year.items()))

        for year, data in numbers_per_year.items():
            # print(str(year) + ": " + str(data[0]) + ", " + str(data[1]) + ", " + str(data[2]) + ", " + str(data[3]))
            if print_total:
                print(data[0])
            else:
                print(data[1])

        print("-----------------------")

    def save_aggregated_encountered_papers(self):

        # Get the paper with the most sections!
        self.encountered_papers_general = EncounteredPapersGeneral("aiide-dataset", 50)

        for paper_name, paper_object in self.aggregated_encountered_papers.items():
            print(paper_name)

            self.encountered_papers_general.add_sections(paper_object.get_paper_sections_raw())

            self.encountered_papers_general.add_full_datapoint(
                paper_object.get_paper_name(),
                paper_object.paper_number,
                paper_object.paper_counter,
                paper_object.get_paper_session(),
                paper_object.get_paper_proceedings(),
                paper_object.keyword
            )

        df = pd.DataFrame(self.encountered_papers_general.get_data(),
                          columns=self.encountered_papers_general.get_data().keys())
        df_cleaned = df.applymap(self.remove_illegal_characters)
        # df.to_excel("output.xlsx")
        # df_cleaned.to_excel(self.papers_directory + "/../aiide-dataset-peryeartest-third-nocontains.xlsx")
        df_cleaned.to_excel(self.papers_directory + "/../results_keywords/aggregated_counters.xlsx")

    def search_references_per_paper(self, key_reference, sentences_to_search: [], all_ngrams_until=True, minimum_length=1,
                                  top_k=5, avg=True,
                                  sd=True, file_name="paper_metrics.csv", save=True, append=True):

        # all_sentence_counts = Counter()
        paper_counter = -1
        self.encountered_papers = {}
        self.encountered_papers_general = {}

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            finished_searching_paper = False
            paper_counter += 1
            paper_already_counted = False

            for section in paper_sections:

                if finished_searching_paper:
                    break

                text = " ".join(section.split())
                text = text.lower()

                # Extract sentences based on ".", ",", ";", ":", "!", "?", and "\n"
                sentences = re.split(r'[.?!:\n]', text)

                # Remove leading and trailing whitespaces from each sentence
                sentences = [sentence.strip() for sentence in sentences]

                # Filter out based on keywords
                sentences = [sentence for sentence in sentences if
                             not self.contains_keyword(sentence, self.ignore_keywords_analysis)]

                for sentence_to_search in sentences_to_search:

                    # Filter out sentences that do not contain the specified substring
                    sentences_searched = [sentence for sentence in sentences if
                                          sentence_to_search.lower() in sentence.lower()]

                    # Filter out sentences that are either one word or empty space
                    sentences_searched = [sentence for sentence in sentences_searched if
                                          len(sentence.split()) > minimum_length]

                    if sentences_searched:

                        current_paper = None

                        if not paper_name in self.encountered_papers:
                            self.encountered_papers.update(
                                {paper_name: EncounteredPapers(paper_name, paper.paper_year, paper.paper_session,
                                                               keyword, paper_counter)})

                        current_paper = self.encountered_papers[paper_name]

                        for found_sentence in sentences_searched:
                            current_paper.add_sentence(found_sentence)

                        if not paper_already_counted:
                            if not paper_name in self.aggregated_encountered_papers:
                                self.aggregated_encountered_papers.update(
                                    {paper_name: EncounteredPapers(paper_name, paper.paper_year, paper.paper_session,
                                                                   keyword, paper_counter)})
                            else:
                                self.aggregated_encountered_papers[paper_name].add_counter()

                            paper_already_counted = True

        self.save_encountered_papers_keywords(keyword)
        self.extract_statistics(keyword, self.encountered_papers, True)

    def search_keywords_per_paper(self, keyword, sentences_to_search: [], all_ngrams_until=True, minimum_length=1, top_k=5, avg=True,
                                    sd=True, file_name="paper_metrics.csv", save=True, append=True):

        # all_sentence_counts = Counter()
        paper_counter = -1
        self.encountered_papers = {}
        self.encountered_papers_general = {}

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            finished_searching_paper = False
            paper_counter += 1
            paper_already_counted = False

            for section in paper_sections:

                if finished_searching_paper:
                    break

                text = " ".join(section.split())
                text = text.lower()

                # Extract sentences based on ".", ",", ";", ":", "!", "?", and "\n"
                sentences = re.split(r'[.?!:\n]', text)

                # Remove leading and trailing whitespaces from each sentence
                sentences = [sentence.strip() for sentence in sentences]

                # Filter out based on keywords
                sentences = [sentence for sentence in sentences if not self.contains_keyword(sentence, self.ignore_keywords_analysis)]

                for sentence_to_search in sentences_to_search:

                    # Filter out sentences that do not contain the specified substring
                    sentences_searched = [sentence for sentence in sentences if sentence_to_search.lower() in sentence.lower()]

                    # Filter out sentences that are either one word or empty space
                    sentences_searched = [sentence for sentence in sentences_searched if len(sentence.split()) > minimum_length]

                    if sentences_searched:

                        current_paper = None

                        if not paper_name in self.encountered_papers:
                            self.encountered_papers.update(
                                {paper_name: EncounteredPapers(paper_name, paper.paper_year, paper.paper_session,
                                                               keyword, paper_counter)})

                        current_paper = self.encountered_papers[paper_name]

                        for found_sentence in sentences_searched:
                            current_paper.add_sentence(found_sentence)

                        if not paper_already_counted:
                            if not paper_name in self.aggregated_encountered_papers:
                                self.aggregated_encountered_papers.update(
                                    {paper_name: EncounteredPapers(paper_name, paper.paper_year, paper.paper_session,
                                                                   keyword, paper_counter)})
                            else:
                                self.aggregated_encountered_papers[paper_name].add_counter()

                            paper_already_counted = True


        self.save_encountered_papers_keywords(keyword)
        self.extract_statistics(keyword, self.encountered_papers, True)


    def search_sentences_per_paper(self, sentences_to_search: [], all_ngrams_until=True, minimum_length=1, top_k=5, avg=True,
                                    sd=True, file_name="paper_metrics.csv", save=True, append=True):

        # all_sentence_counts = Counter()

        all_sentence_counts = {x: Counter() for x in sentences_to_search}

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            seen_sentence = {x: set() for x in sentences_to_search}

            for section in paper_sections:

                # Preprocess the text (remove non-alphanumeric characters, convert to lowercase)
                text = re.sub(r'[^a-zA-Z.,;?!:\s\n\[\]\d-]', '', section)

                # Remove everything inside square brackets
                text = re.sub(r'\[.*?\]', '', text)

                text = re.sub(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]', '', text)

                # Remove punctuation even if there is a space in between, excluding "-"
                text = re.sub(r'([.,;])\s*(?<!\n)', r'\1', text)

                # Replace consecutive spaces with a single space, excluding "\n"
                # text = re.sub(r'(?<!\n)\s+', ' ', text)
                # text = re.sub(r'(?<!\n)\s+(?!\n)', ' ', text)

                # Remove spaces before punctuation, excluding "-"
                # text = re.sub(r'\s*([.,;?!:])\s*', r'\1', text)
                # text = re.sub(r'\s*([.,;?!:])', r'\1', text)
                text = re.sub(r'\s*([.,;?!:])(?<!\n)', r'\1', text)

                text = text.lower()

                # Extract sentences based on ".", ",", ";", and "\n"
                # sentences = re.split(r'[.,;\n]', text)

                # Extract sentences based on ".", ",", ";", ":", "!", "?", and "\n"
                sentences = re.split(r'[.,;?!:\n]', text)

                # Remove leading and trailing whitespaces from each sentence
                sentences = [sentence.strip() for sentence in sentences]

                # Filter out based on keywords
                sentences = [sentence for sentence in sentences if not self.contains_keyword(sentence, self.ignore_keywords_analysis)]

                for sentence_to_search in sentences_to_search:
                    # Filter out sentences that do not contain the specified substring
                    sentences_searched = [sentence for sentence in sentences if sentence_to_search.lower() in sentence.lower()]

                    # Filter out sentences that are either one word or empty space
                    sentences_searched = [sentence for sentence in sentences_searched if len(sentence.split()) > minimum_length]

                    # Count each unique sentence
                    sentence_counts = Counter(sentences_searched)

                    for sentence_count in sentence_counts:
                        # Count only if the n-gram hasn't been seen in this text before
                        if sentence_count not in seen_sentence[sentence_to_search]:
                            all_sentence_counts[sentence_to_search][sentence_count] += 1
                            # all_ngram_counts[ngram] += 1
                            seen_sentence[sentence_to_search].add(sentence_count)

        # Display the top 5 n-grams by counts sorted!
        print("Here we present all sentences occurrences in papers")
        total_sentences = len(self.papers)

        for key, all_sentence_per_word in all_sentence_counts.items():
            print(f"Here we present all sentences occurrences in papers when using search string: '{key}' with minimum length {minimum_length}")
            sum_count = 0
            for sentence, count in sorted(all_sentence_per_word.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                percentage = (count / total_sentences) * 100
                print(f"{sentence}: {count} occurrences ({percentage:.2f}% of total)")
                sum_count += count

            percentage = (sum_count / total_sentences) * 100
            print(f"\n the sum of ocurrences is {sum_count} out of {total_sentences}. This means ({percentage:.2f}% of total)")

            print("\n")

    def remove_punctuation(self, input_string):
        # Make a regular expression that matches all punctuation
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub('', input_string)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        # Use the regex
        return text

    def count_occurrences_of_string(self, target_string):
        total_occurrences = 0

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            paper_sections_no_punctuation = [self.remove_punctuation(x) for x in paper_sections]

            for section in paper_sections_no_punctuation:

                occurrences = section.count(target_string.lower())
                total_occurrences += occurrences


        for text in texts:
            # Case-insensitive search
            occurrences = text.lower().count(target_string.lower())
            total_occurrences += occurrences

        return total_occurrences

    def create_limited_dataset_text_references(self, input_type, token_limit=512, section_token_limit=400):
        dataset_llm = DatasetLLMPapers()

        for paper_name, paper in self.papers.items():
            paper_sections = paper.get_paper_sections_no_delimiter()
            paper_section_titles = paper.get_paper_section_titles_no_delimiter()
            paper_references = paper.get_paper_references_no_delimiter()

            for i in range(0, len(paper_sections)):

                dataset_llm.format_data_section_paragraphs(input_type, paper_section_titles[i], paper_sections[i],
                                                       paper_references, paper.get_paper_session(), section_token_limit,
                                                       token_limit, True)


            print(paper.get_paper_name())

        df = pd.DataFrame(dataset_llm.get_data(),
                          columns=dataset_llm.get_data().keys())
        # df.to_excel("output.xlsx")
        df.to_excel(self.papers_directory + "/chi-2023-data-no-references-2048-parts.xlsx")
        # df.to_excel(self.papers_directory + "/chi-2023-data-no-references.xlsx") # Next test


    def papers_to_compressed_view(self):

        # Get the paper with the most sections!
        self.collected_papers_general = CollectedPapersGeneral("chi23", self.max_sections)

        for paper_name, paper_object in self.papers.items():
            print(paper_name)

            self.collected_papers_general.add_sections(paper_object.get_paper_sections_raw())

            self.collected_papers_general.add_full_datapoint(
                paper_object.get_paper_name(),
                paper_object.get_paper_session(),
                paper_object.get_paper_abstract(),
                paper_object.get_paper_authors(),
                paper_object.get_paper_full_authors(),
                paper_object.get_paper_section_titles(),
                # paper_object.get_paper_sections(),
                paper_object.get_paper_references()
            )

        df = pd.DataFrame(self.collected_papers_general.get_data(), columns=self.collected_papers_general.get_data().keys())
        #df.to_excel("output.xlsx")
        df.to_excel(self.papers_directory + "/compressed_view.xlsx")
