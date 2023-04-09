from tika import parser
import re
import pytesseract
import pdfplumber
from bidi import algorithm as bidialg
import json
from pathlib import Path

PATTERN_Q_A_NUMBER = "(\\d+)(\\)|\\.)"

PATTERN_Q_HEB = "שאלה (מס|מספר)"

PATTERN_A_HEB = "(תשובה|התשובה|תשובות)(|:)"


def get_using_tika():
    with open("./QA_bible.txt", "w", encoding="utf-8") as f:
        raw = parser.from_file("english.nationalexam2020-1.pdf")
        list_questions = re.findall(f"{PATTERN_Q_A_NUMBER}(.*?\\?)(.*?)(?={PATTERN_Q_A_NUMBER})", raw["content"],
                                    flags=re.DOTALL)
        for qa in list_questions:
            question = qa[2]
            answers = qa[3]
            question_formatted = re.sub("\n+", " ", question)
            question_formatted = re.sub(" +", " ", question_formatted)
            question_formatted = question_formatted.replace("\u200b", "")
            answers_formatted = re.sub("\n+", "\n", answers)
            answers_formatted = re.sub(" +", " ", answers_formatted)
            answers_formatted = re.sub("([a-zA-Z]( )*(\\.|\\)))", "", answers_formatted).split("\n")
            print(f"question: {question_formatted}")
            f.write(f"question: {question_formatted}\n")
            answers_formatted = ",".join(
                [answer.strip().replace("\u200b", "") for answer in answers_formatted if answer.strip() != ''])
            print(f"answers: [{answers_formatted}]")
            f.write(f"answers: [{answers_formatted}]\n")


def get_using_pdf2go_conversion_and_OCR():
    dict_questions_and_answers = {}
    with open("Q&A.txt", "w", encoding="utf-8") as f:
        for file_name in Path("quizes").glob("**/*.txt"):
            print(file_name)
            page_text = bidialg.get_display(open(file_name ,"w").read())
            f.write(page_text)
            list_questions_and_answers = re.findall(f"({PATTERN_Q_HEB} \\d+)(.*)", page_text, flags=re.DOTALL)
            for question_and_answer in list_questions_and_answers:
                try:
                    question_header_number = question_and_answer[0]
                    question_body = question_and_answer[2]
                    list_questions_and_answers_single_question = re.search(f"^(.*?){PATTERN_A_HEB}(.*?)$",
                                                                           question_body, re.DOTALL)
                    questions = list_questions_and_answers_single_question.group(1)
                    answers = list_questions_and_answers_single_question.group(4)
                    question_prolog = re.search(f"^(.*?)(?=\\s+?[\u05D0 -\u05D1]\\..*\\s*?)", question_body,
                                                re.DOTALL).group(1)
                    questions_list = [str(question).replace("\n", "") for question in
                                      re.findall("(\\s+?[\u05D0 -\u05D1]\\..*\\s*?)", questions)]
                    answers_list = [str(answer).replace("\n", "") for answer in
                                    re.findall("(\\s+[\u05D0 -\u05D1]\\..*\\s+)", answers)]
                    dict_questions_and_answers[question_header_number + "_" + file_name.name] = {
                        "prolog": str(question_prolog).replace("\n", ""),
                        "questions": questions_list,
                        "answers": answers_list}
                except Exception as e:
                    print(e)
    with open("Q&A.json", "w", encoding="utf-8") as json_f:
        json.dump(dict_questions_and_answers, json_f, ensure_ascii=False, indent=4)


def get_using_pdfplumber():
    dict_questions_and_answers = {}
    with open("Q&A.txt", "w", encoding="utf-8") as f:
        for file_name in Path("quizes").glob("*.pdf"):
            with pdfplumber.open(file_name) as pdf:
                print(file_name)
                for i in range(len(pdf.pages)):
                    pdf_page = pdf.pages[i].dedupe_chars()
                    page_text = bidialg.get_display(pdf_page.extract_text())
                    f.write(page_text)
                    list_questions_and_answers = re.findall(f"({PATTERN_Q_HEB} \\d+)(.*)", page_text, flags=re.DOTALL)
                    for question_and_answer in list_questions_and_answers:
                        try:
                            question_header_number = question_and_answer[0]
                            question_body = question_and_answer[2]
                            list_questions_and_answers_single_question = re.search(f"^(.*?){PATTERN_A_HEB}(.*?)$",
                                                                                   question_body, re.DOTALL)
                            questions = list_questions_and_answers_single_question.group(1)
                            answers = list_questions_and_answers_single_question.group(4)
                            question_prolog = re.search(f"^(.*?)(?=\\s+?[\u05D0 -\u05D1]\\..*\\s*?)", question_body,
                                                        re.DOTALL).group(1)
                            questions_list = [str(question).replace("\n", "") for question in
                                              re.findall("(\\s+?[\u05D0 -\u05D1]\\..*\\s*?)", questions)]
                            answers_list = [str(answer).replace("\n", "") for answer in
                                            re.findall("(\\s+[\u05D0 -\u05D1]\\..*\\s+)", answers)]
                            dict_questions_and_answers[question_header_number + "_" + file_name.name] = {
                                "prolog": str(question_prolog).replace("\n", ""),
                                "questions": questions_list,
                                "answers": answers_list}
                        except Exception as e:
                            print(e)
    with open("Q&A.json", "w", encoding="utf-8") as json_f:
        json.dump(dict_questions_and_answers, json_f, ensure_ascii=False, indent=4)


def get_using_google_OCR():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    print(pytesseract.get_languages(config=''))


def main():
    get_using_google_OCR()


if __name__ == "__main__":
    main()
