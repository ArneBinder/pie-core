# The implementations in this file are taken from the PyTorch Lightning codebase.
# Here is the original license header:
#
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pie_core import PieHyperparametersMixin


def test_save_hyperparameters_under_composition():
    """Test that in a composition where the parent is not a Lightning-like module, the parent's
    arguments don't get collected."""

    class ChildInComposition(PieHyperparametersMixin):
        def __init__(self, same_arg):
            super().__init__()
            self.save_hyperparameters()

    class NotPLSubclass:  # intentionally not subclassing LightningModule/LightningDataModule
        def __init__(self, same_arg="parent_default", other_arg="other"):
            self.child = ChildInComposition(same_arg="cocofruit")

    parent = NotPLSubclass()
    assert parent.child.hparams == {"same_arg": "cocofruit"}


def test_save_hyperparameters_ignore():
    """Test if `save_hyperparameter` applies the ignore list correctly during initialization."""

    class PLSubclass(PieHyperparametersMixin):
        def __init__(self, learning_rate=1e-3, optimizer="adam"):
            super().__init__()
            self.save_hyperparameters(ignore=["learning_rate"])

    pl_instance = PLSubclass(learning_rate=0.01, optimizer="sgd")
    assert pl_instance.hparams == {"optimizer": "sgd"}


def test_save_hyperparameters_ignore_under_composition():
    """Test that in a composed system, hyperparameter saving skips ignored fields from nested
    modules."""

    class ChildModule(PieHyperparametersMixin):
        def __init__(self, dropout, activation, init_method):
            super().__init__()
            self.save_hyperparameters(ignore=["dropout", "activation"])

    class ParentModule(PieHyperparametersMixin):
        def __init__(self, batch_size, optimizer):
            super().__init__()
            self.child = ChildModule(dropout=0.1, activation="relu", init_method="xavier")

    class PipelineWrapper:  # not a Lightning subclass on purpose
        def __init__(self, run_id="abc123", seed=42):
            self.parent_module = ParentModule(batch_size=64, optimizer="adam")

    pipeline = PipelineWrapper()
    assert pipeline.parent_module.child.hparams == {
        "init_method": "xavier",
        "batch_size": 64,
        "optimizer": "adam",
    }
