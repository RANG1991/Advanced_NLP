from tika import parser
import re


PATTERN_Q_A_NUMBER = "(\\d+)(\\)|\\.)"


def main():
    with open("./QA_bible.txt", "w", encoding="utf-8") as f:
        raw = parser.from_file("english.nationalexam2020-1.pdf")
        list_questions = re.findall(f"{PATTERN_Q_A_NUMBER}(.*?\\?)(.*?)(?={PATTERN_Q_A_NUMBER})", raw["content"], flags=re.DOTALL)
        for qa in list_questions:
            question = qa[2]
            answers = qa[3]
            question_formatted = re.sub("\n+", " ", question)
            question_formatted = re.sub(" +", " ", question_formatted)
            question_formatted = question_formatted.replace("\u200b", "")
            answers_formatted = re.sub("\n+", " ", answers)
            answers_formatted = re.sub(" +", " ", answers_formatted)
            answers_formatted = re.sub("([a-zA-Z]( )*(\\.|\\)))", "", answers_formatted)
            print(f"question: {question_formatted}")
            f.write(f"question: {question_formatted}\n")
            answers_formatted = [answer.strip().replace("\u200b", "") for answer in answers_formatted.split('\n')
                                 if answer.strip() != '']
            print(f"answers: {answers_formatted}")
            f.write(f"answers: {answers_formatted}\n")


if __name__ == "__main__":
    main()
