from automarl.components.fundamentals.translator.tensor_translator import ToTorchTranslator
from automarl.components.fundamentals.translator.torch_image_state_translator import ImageNormalizer, ImageReverter, ImageReverterToSingleChannel, ImageSingleChannel
from automarl.components.fundamentals.translator.translator import TranslatorSequence
import numpy as np
import torch
from automarl.components.basic_components import loop_components
from automarl.component import Component, ParameterSignature
import unittest
              
    
class TestToTorchTranslator(unittest.TestCase):

    def test_translate_numpy_to_torch(self):
        state = np.zeros((4, 5, 3), dtype=np.float32)

        translator = ToTorchTranslator(input={
            "original_shape": state.shape
        })
        translator.process_input()

        out = translator.translate_state(state)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, torch.Size([4, 5, 3]))
        self.assertEqual(out.dtype, torch.float32)




class TestImageReverter(unittest.TestCase):

    def setUp(self):
        self.state = torch.zeros((4, 5, 3), dtype=torch.float32)

    def test_shape_conversion(self):
        translator = ImageReverter(input={
            "original_shape": self.state.shape
        })
        translator.process_input()

        self.assertEqual(translator.get_shape(), torch.Size([3, 4, 5]))

    def test_translate_no_buffer(self):
        translator = ImageReverter(input={
            "buffered_operations": False
        })
        translator.process_input()

        out = translator.translate_state(self.state)

        self.assertEqual(out.shape, torch.Size([3, 4, 5]))
        self.assertTrue(torch.equal(out, self.state.permute(2, 0, 1)))

    def test_translate_with_buffer(self):
        translator = ImageReverter(input={
            "original_shape": self.state.shape,
            "buffered_operations": True
        })
        translator.process_input()

        out1 = translator.translate_state(self.state)
        out2 = translator.translate_state(self.state)

        self.assertIs(out1, out2)  # same buffer
        self.assertEqual(out1.shape, torch.Size([3, 4, 5]))


class TestImageSingleChannel(unittest.TestCase):

    def test_single_channel(self):
        state = torch.randn((3, 4, 5))

        translator = ImageSingleChannel(input={
            "original_shape": state.shape
        })
        translator.process_input()

        out = translator.translate_state(state)

        self.assertEqual(out.shape, torch.Size([4, 5]))
        self.assertTrue(torch.equal(out, state[0]))

    def test_shape(self):
        translator = ImageSingleChannel(input={
            "original_shape": (3, 4, 5)
        })
        translator.process_input()

        self.assertEqual(translator.get_shape(), torch.Size([4, 5]))


class TestImageNormalizer(unittest.TestCase):

    def test_out_of_place(self):
        state = torch.ones((4, 5)) * 255

        translator = ImageNormalizer(input={
            "in_place_translation": False
        })
        translator.process_input()

        out = translator.translate_state(state)

        self.assertTrue(torch.allclose(out, torch.ones((4, 5))))
        self.assertFalse(out is state)

    def test_in_place(self):
        state = torch.ones((4, 5)) * 255

        translator = ImageNormalizer(input={
            "in_place_translation": True
        })
        translator.process_input()

        out = translator.translate_state(state)

        self.assertTrue(out is state)
        self.assertTrue(torch.allclose(state, torch.ones((4, 5))))

class TestImageReverterToSingleChannel(unittest.TestCase):

    def setUp(self):
        self.state = torch.randn((4, 5, 3))

    def test_shape(self):
        translator = ImageReverterToSingleChannel(input={
            "original_shape": self.state.shape
        })
        translator.process_input()

        self.assertEqual(translator.get_shape(), torch.Size([4, 5]))

    def test_translate_no_buffer(self):
        translator = ImageReverterToSingleChannel(input={
            "buffered_operations": False
        })
        translator.process_input()

        out = translator.translate_state(self.state)

        self.assertEqual(out.shape, torch.Size([4, 5]))
        self.assertTrue(torch.equal(out, self.state[:, :, 0]))

class TestTranslatorSequence(unittest.TestCase):

    def test_sequence_shape(self):
        seq = TranslatorSequence(input={
            "original_shape": (4, 5, 3),
            "translators_sequence": [
                ImageReverter(input={}),
                ImageSingleChannel(input={})
            ]
        })

        seq.process_input()

        self.assertEqual(seq.get_shape(), torch.Size([4, 5]))

    def test_sequence_translation(self):
        state = torch.randn((4, 5, 3))

        seq = TranslatorSequence(input={
            "translators_sequence": [
                ImageReverter(input={}),
                ImageSingleChannel(input={})
            ]
        })

        seq.process_input()

        out = seq.translate_state(state)

        expected = state.permute(2, 0, 1)[0]
        self.assertTrue(torch.equal(out, expected))

    def test_propagates_in_place_flag(self):
        seq = TranslatorSequence(input={
            "in_place_translation": True,
            "translators_sequence": [
                ImageNormalizer(input={})
            ]
        })

        seq.process_input()

        self.assertTrue(seq.translators_sequence[0].in_place_translation)

if __name__ == '__main__':
    unittest.main()