import contextlib
import io
import unittest
from unittest import mock

from moduleio import cli
from moduleio.predictor import PredictionError


class CLIContractTests(unittest.TestCase):
    def test_expected_inference_failure_is_a_cli_error_without_traceback(self):
        arguments = ["predict", "--checkpoint", "weights.pt", "--module", "example",
                     "--hook", "respond", "--arguments", "input.json"]
        stderr = io.StringIO()
        with mock.patch.object(cli, "_predict", side_effect=PredictionError("missing EOS")):
            with contextlib.redirect_stderr(stderr):
                result = cli.main(arguments)
        self.assertEqual(result, 2)
        self.assertIn("missing EOS", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
