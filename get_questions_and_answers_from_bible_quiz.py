from tika import parser
import re


PATTERN_Q_A_NUMBER = "(\\d+)(\\)|\\.)"


def main():
    raw = parser.from_file("english.nationalexam2020-1.pdf")
    list_questions = re.findall(f"{PATTERN_Q_A_NUMBER}(.*?\\?)(.*?)(?={PATTERN_Q_A_NUMBER})", raw["content"], flags=re.DOTALL)
    for qa in list_questions:
        question = qa[2]
        answers = qa[3]
        question_formatted = re.sub("\n+", "\n", question)
        question_formatted = re.sub(" +", " ", question_formatted)
        answers_formatted = re.sub("\n+", "\n", answers)
        answers_formatted = re.sub(" +", " ", answers_formatted)
        answers_formatted = re.sub("([a-zA-Z]( )*(\\.|\\)))", "", answers_formatted)
        print("question: ", question_formatted)
        print("answers:", [answer.strip() for answer in answers_formatted.split("\n") if answer.strip() != ""])


if __name__ == "__main__":
    main()
