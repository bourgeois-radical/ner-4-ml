from typing import List, Tuple
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize


class NamedEntities:

    @staticmethod
    def initialize_labels(src_path: str, dst_path: str) -> str:

        # transform str path to WindowsPath
        dst_path = Path(dst_path)

        # get the path of each and every pdf file
        pointer_txt_paths = Path(src_path).glob('**/*')
        all_txt_paths = [x for x in pointer_txt_paths if x.is_file()]

        for txt_path in all_txt_paths:
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                paper_text = txt_file.read()
                split_paper_text = paper_text.splitlines()
                paper_text = ' '.join(split_paper_text)

            # tokenize by sentence
            paper_text_tokenized_by_sent = sent_tokenize(paper_text)

            # tokenize every sent by word
            paper_text_sent_tokenized_by_word = []
            for sent in paper_text_tokenized_by_sent:
                paper_text_sent_tokenized_by_word.append(word_tokenize(sent))

            # initialize machine learning ner tokens
            ml_ner_labels = []
            for sent in paper_text_sent_tokenized_by_word:
                ml_ner_labels.append([0]*len(sent))

            # write sentences with initialized NER labels into .txt file for further annotation
            sentns_with_ner_labels = []
            for sent, labels in zip(paper_text_sent_tokenized_by_word, ml_ner_labels):
                one_sent_with_ner_labels = []
                for token, ner_label in zip(sent, labels):
                    one_sent_with_ner_labels.append((token, ner_label))

                sentns_with_ner_labels.append(one_sent_with_ner_labels)

            dst = str(dst_path / txt_path.name.replace('.txt', '')) + '_ml_ner' + '.txt'

            with open(dst, 'w', encoding='utf-8') as txt_file:
                for sent_num, sent_with_ner_tags in enumerate(sentns_with_ner_labels):
                    # write each item on a new line
                    enumerated_sent = str(sent_num) + ' ' + str(sent_with_ner_tags)
                    txt_file.write("%s\n\n" % enumerated_sent)

        return 'Done'

    @staticmethod
    def build_dataset(src_path: str) -> Tuple[List[str], List[int]]:
        """Распарсить проаннотированные данные"""

        pointer_pdf_paths = Path(src_path).glob('**/*')
        all_paths = [x for x in pointer_pdf_paths if x.is_file()]

        # first: collect all tokens, labels and tags for one paper
        one_tokenized_paper = []
        one_labeled_paper = []
        one_tagged_paper = []

        # second: collect all papers' tokens, labels and tags like:
        # [[first_paper_tokens], [second_paper_tokens] ... etc.]
        # [[first paper_labels], [second_paper_labels] ... etc.]
        all_tokenized_papers = []
        all_labeled_papers = []
        all_tagged_papers = []

        for idx_txt, annotated_ner_txt in enumerate(all_paths):
            with open(annotated_ner_txt, 'r', encoding='utf-8') as txt_file:
                paper_text = txt_file.read()
                all_sentences = paper_text.splitlines()

            # get two lists: tokens and their labels (0 or 1)
            for sent in all_sentences:

                # 44 [('Prediction', 0), ('has', 0), ('been', 0), ('made', 0), ('on', 0)]
                if len(sent) != 0 and sent.count('[') == 1:

                    # delete '[' with the sentence number and ']'
                    sent = sent[:-1].split('[')[1]

                    # get pairs "(token, label"
                    token_label_pairs = sent.split('),')

                    # print(f"{annotated_ner_txt}: {token_label_pairs} \n")

                    # get clean pairs without "("
                    tokens = [x.split("',")[0].strip().replace("('", "") for x in token_label_pairs
                              if len(x.split("',")[1]) != 0]

                    # get labels
                    labels = [int(x.split("',")[1].replace(')', '')) for x in token_label_pairs
                              if len(x.split("',")[1]) != 0]

                    # collect all tokens
                    one_tokenized_paper.append(tokens)

                    # collect all tokens' labels
                    one_labeled_paper.append(labels)

            # we have to distinguish between the Beginning and the Inside of an Entity
            # ['5.4',
            #  'Ensemble', # = 1 = B-ML
            #  'of',       # = 2 = I-ML
            #  'trees',    # = 2 = I-ML
            # so: # labels = (0, 1, 2); tags = ('O', 'B-ML', 'I-ML')
            # the following code is responsible for

            # this map function will be applied later in the loop
            def map_to_tags(x: int) -> str:
                """Maps (0, 1, 2) to ('O', 'B-ML', 'I-ML')"""
                if x == 0:
                    return 'O'
                elif x == 1:
                    return 'B-ML'
                elif x == 2:
                    return 'I-ML'

            inside_entity = False
            b_ml_idx = 0
            for idx_snt, sentence in enumerate(one_labeled_paper):
                for idx_lbl, label in enumerate(sentence):
                    # the beginning of an entity
                    if (idx_lbl + 1) < len(sentence) and label == 0 and sentence[idx_lbl+1] == 1:  #  wanted to use "next"
                        # don't need to change label at the beginning
                        inside_entity = True
                        # remember the position of the first token of an entity
                        b_ml_idx = idx_lbl + 1
                    # inside an entity
                    elif inside_entity and idx_lbl != b_ml_idx and label != 0:
                        one_labeled_paper[idx_snt][idx_lbl] = 2  # 2 = I-ML

                # get list of tags
                sent_with_ml_tags = list(map(map_to_tags, sentence))
                one_tagged_paper.append(sent_with_ml_tags)

            # second: collect all papers' tokens, labels and tags like:
            # [[first_paper_tokens], [second_paper_tokens] ... etc.]
            # [[first paper_labels], [second_paper_labels] ... etc.]
            all_tokenized_papers.append(one_tokenized_paper)
            all_labeled_papers.append(one_labeled_paper)
            all_tagged_papers.append(one_tagged_paper)

            one_tokenized_paper = []
            one_labeled_paper = []
            one_tagged_paper = []

        # let's build the entire dataset as a dictionary
        dataset = {
            'papers_tokens': all_tokenized_papers,
            'papers_labels': all_labeled_papers,
            'papers_tags': all_tagged_papers
        }

        print('The dataset has been successfully built!')

        return dataset







