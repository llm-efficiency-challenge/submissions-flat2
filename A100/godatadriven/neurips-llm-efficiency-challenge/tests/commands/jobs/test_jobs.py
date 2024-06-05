from typer.testing import CliRunner

from cajajejo.commands.jobs.evaluation import evaluation_cmd


runner = CliRunner()


def test_validate_config(evaluation_config_on_disk):
    result = runner.invoke(
        evaluation_cmd,
        ["validate-job-config", evaluation_config_on_disk],
    )
    assert result.exit_code == 0
