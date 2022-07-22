"""
TODO: Refactoring needed
- out of use, at the moment (evaluation files were refactored)

Together with the MardownTestResult class, this module provides a clean beautiful way to compare runs
MardownTestResults need to be created in the experiement with the right functions from analytics.py
"""

from evaluation.MarkdownTestResult import MarkdownTestResult
from models.RainbowModel import RainbowModel
from utils.markdown import markdown_table_str
import statistics

from utils.telegram import send_telegram
from utils import settings
import os

# General --------------------------------------------------------------------------------------------------------------


class MarkdownReport:

    def create_send_save(
        self,
        title: str,
        description: str,
        models_evaluation_result: "list[MarkdownTestResult]",
        has_context_accuracy=False,
        telegram: bool = True,
    ) -> None:
        """
        list[MarkdownTestResult]
        """
        report_str = ""

        # comparison table
        comparison_table: "list[list[str | int | float]]" = [
            ["", "correct_classification_acc", "avg_failure_rate"]
        ]
        for model_evaluation_result in models_evaluation_result:
            cor_class_acc = round(
                model_evaluation_result.correct_classification_accuracy, ndigits=2
            )
            avg_fail_rate = round(
                model_evaluation_result.average_failure_rate, ndigits=2
            )
            comparison_table.append(
                [
                    'Model "'
                    + model_evaluation_result.model_nickname
                    + '" @ '
                    + str(model_evaluation_result.model.kwargs),
                    cor_class_acc,
                    avg_fail_rate,
                ]
            )

        if has_context_accuracy:
            comparison_table[0].append("context_accuracy")
            for i in range(len(models_evaluation_result)):
                comparison_table[i + 1].append(
                    str(models_evaluation_result[i].context_accuracy)
                )

        report_str += markdown_table_str(comparison_table)

        self._create_rainbow_report(title, description, report_str)

        if telegram:
            self._send_telegram_report(title, description, report_str)

    def create_send_save_kfold(
        self,
        title: str,
        description: str,
        models_evaluation_results: "list[list[MarkdownTestResult]]",
        telegram: bool = True,
    ) -> None:
        """
        list[list[MarkdownTestResult]]
        """

        report_str = ""

        # comparison table
        comparison_table: "list[list[str | int | float]]" = [
            [""],
            ["correct_classification_acc"],
            ["avg_failure_rate"],
        ]
        for i in range(len(models_evaluation_results)):
            current_model_results = models_evaluation_results[i]
            current_model_nickname = current_model_results[0].model_nickname

            comparison_table[0].append('Model "' + current_model_nickname + '"')
            comparison_table[1].append(
                round(
                    sum(
                        [
                            model_evaluation_result.correct_classification_accuracy
                            for model_evaluation_result in current_model_results
                        ]
                    )
                    / len(current_model_results),
                    ndigits=2,
                )
            )
            comparison_table[2].append(
                round(
                    sum(
                        [
                            model_evaluation_result.average_failure_rate
                            for model_evaluation_result in current_model_results
                        ]
                    )
                    / len(current_model_results),
                    ndigits=2,
                )
            )

        report_str += markdown_table_str(comparison_table)

        # k_fold_evaluation
        for model_evaluation_results in models_evaluation_results:
            report_str += self._k_fold_report_str(model_evaluation_results)

        self._create_rainbow_report(title, description, report_str)

        if telegram:
            self._send_telegram_report(title, description, report_str)

    def _create_rainbow_report(self, title: str, description: str, report: str) -> None:
        base_path = os.path.join(settings.ML_RAINBOW_PATH, "rainbow_test/report/")
        path = os.path.join(base_path, title + ".md")
        if os.path.exists(path):
            num = 0
            while os.path.exists(path):
                num += 1
                path = os.path.join(base_path, title + "_" + str(num) + ".md")

        with open(path, "w") as f:
            f.write("# " + title + "\n" + description + "\n\n\n" + report)

    def _send_telegram_report(self, title: str, description: str, report: str) -> None:
        send_telegram("# " + title + "\n" + description + "\n\n\n" + report)

    # Specific Report str --------------------------------------------------------------------------------------------------------------

    def _k_fold_table_str(self, test_reports: "list[MarkdownTestResult]") -> str:
        markdown_array = []
        markdown_array.append(
            [
                "k_fold_idx",
                "correct_classification_acc",
                "avg_failure_rate",
                "test_activity_distribution",
            ]
        )

        correct_classification_accuracies = [
            round(report.correct_classification_accuracy, ndigits=2)
            for report in test_reports
        ]
        avg_failure_rates = [
            round(report.average_failure_rate, ndigits=2) for report in test_reports
        ]

        for i in range(len(test_reports)):
            markdown_array.append(
                [
                    i,
                    correct_classification_accuracies[i],
                    avg_failure_rates[i],
                    test_reports[i].test_activity_distribution,
                ]
            )

        markdown_array.append(["", "", "", ""])
        markdown_array.append(
            ["min", min(correct_classification_accuracies), min(avg_failure_rates), "-"]
        )
        markdown_array.append(
            ["max", max(correct_classification_accuracies), max(avg_failure_rates), "-"]
        )
        markdown_array.append(
            [
                "mean",
                round(sum(correct_classification_accuracies) / len(test_reports), 2),
                round(sum(avg_failure_rates) / len(test_reports), 2),
                "-",
            ]
        )
        markdown_array.append(
            [
                "median",
                statistics.median(correct_classification_accuracies),
                statistics.median(avg_failure_rates),
                "-",
            ]
        )

        return markdown_table_str(markdown_array)

    def _k_fold_report_str(self, evaluation_results: "list[MarkdownTestResult]") -> str:
        """
        Model1 - nickname
        {kwargs}
        K_fold_table with mean and average
        """
        report = ""

        # Model Specification
        report += (
            '### Model "'
            + evaluation_results[0].model_nickname
            + '"\n\n'
            + str(evaluation_results[0].model.kwargs)
            + "\n\n"
        )
        report += self._k_fold_table_str(evaluation_results)
        return report

    @staticmethod
    def markdown_table_str(input_list: list) -> str:

        """
        Input: Python list with rows of table as lists
                First element as header.
            Output: String to put into a .md file

        Ex Input:
            [["Name", "Age", "Height"],
            ["Jake", 20, 5'10],
            ["Mary", 21, 5'7]]
        """

        markdown = "\n" + str("| ")

        for e in input_list[0]:
            to_add = " " + str(e) + str(" |")
            markdown += to_add
        markdown += "\n"

        markdown += "|"
        for i in range(len(input_list[0])):
            markdown += str("-------------- | ")
        markdown += "\n"

        for entry in input_list[1:]:
            markdown += str("| ")
            for e in entry:
                to_add = str(e) + str(" | ")
                markdown += to_add
            markdown += "\n"

        return markdown + "\n"

