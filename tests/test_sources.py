import unittest
from unittest.mock import Mock,patch
from answer_generator import AnswerGenerator

class SourceTests(unittest.TestCase):
    def test_distinct_verses_and_hadith_references_survive(self):
        g=AnswerGenerator.__new__(AnswerGenerator)
        docs=[{'source_type':'quran','surah':1,'ayah_start':a,'ayah_end':a,'score':1} for a in [1,2]]
        docs += [{'source_type':'hadith','book':'book','hadith_number':n,'source_url':f'https://example.org/{n}'} for n in [3,4]]
        sources=g.extract_sources(docs)
        self.assertEqual(len(sources),4)
        self.assertEqual([s['ayah_start'] for s in sources[:2]],[1,2])
        self.assertEqual([s['hadith_number'] for s in sources[2:]],[3,4])

    def test_default_answer_is_unverified_and_calls_no_api(self):
        g=AnswerGenerator.__new__(AnswerGenerator); g.client=Mock()
        answer=g.generate_answer('سؤال',[{'clean_text':'نص اختبار غير ديني','source_type':'test'}])
        self.assertFalse(answer['verified'])
        self.assertIn('نص اختبار غير ديني',answer['answer'])
        g.client.responses.create.assert_not_called()

if __name__=='__main__': unittest.main()
