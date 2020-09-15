from unittest import TestCase
from src.main.python.edu.arizona.cs.invertedindex import InvertedIndex

class TryTesting(TestCase):
    docs="src/main/resources/Docs.txt"

    def test_q5_1(self):
        query_qn5_1 = "schizophrenia AND drug"
        ans_qn5_1=InvertedIndex(self.docs).q5_1(query_qn5_1)
        assert type(ans_qn5_1) is not None
        assert type(ans_qn5_1) is list
        assert len(ans_qn5_1) >0

        assert (ans_qn5_1[0]) is not None
        assert (type(ans_qn5_1[0])) is str
        assert (ans_qn5_1[0]) == "Doc1"

        assert (ans_qn5_1[1]) is not None
        assert (type(ans_qn5_1[1])) is str
        assert (ans_qn5_1[1]) == "Doc4"

    def test_q5_2(self):
        query_qn5_2="breakthrough OR new"
        ans_qn5_2 = InvertedIndex(self.docs).q5_2(query_qn5_2)
        assert type(ans_qn5_2) is not None
        assert type(ans_qn5_2) is list
        assert len(ans_qn5_2) > 0

        assert (ans_qn5_2[0]) is not None
        assert (type(ans_qn5_2[0])) is str
        assert (ans_qn5_2[0]) == "Doc1"

        assert (ans_qn5_2[1]) is not None
        assert (type(ans_qn5_2[1])) is str
        assert (ans_qn5_2[1]) == "Doc2"

        assert (ans_qn5_2[2]) is not None
        assert (type(ans_qn5_2[2])) is str
        assert (ans_qn5_2[2]) == "Doc3"

        assert (ans_qn5_2[3]) is not None
        assert (type(ans_qn5_2[3])) is str
        assert (ans_qn5_2[3]) == "Doc4"

    def test_q5_3(self):
        query_qn5_3 = "(drug OR treatment) AND schizophrenia"
        ans_qn5_3 = InvertedIndex(self.docs).q5_3(query_qn5_3)
        assert type(ans_qn5_3) is not None
        assert type(ans_qn5_3) is list
        assert len(ans_qn5_3) > 0

        assert (ans_qn5_3[0]) is not None
        assert (type(ans_qn5_3[0])) is str
        assert (ans_qn5_3[0]) == "Doc1"

        assert (ans_qn5_3[1]) is not None
        assert (type(ans_qn5_3[1])) is str
        assert (ans_qn5_3[1]) == "Doc2"

        assert (ans_qn5_3[2]) is not None
        assert (type(ans_qn5_3[2])) is str
        assert (ans_qn5_3[2]) == "Doc4"

