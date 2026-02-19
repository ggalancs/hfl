# SPDX-License-Identifier: HRUL-1.0
"""Tests for model license verification and classification."""

from unittest.mock import MagicMock, patch

from hfl.hub.license_checker import (
    LICENSE_CLASSIFICATION,
    LICENSE_RESTRICTIONS,
    LicenseInfo,
    LicenseRisk,
    check_model_license,
    get_license_summary,
    require_user_acceptance,
)


class TestLicenseRisk:
    """Tests for LicenseRisk enum."""

    def test_all_risk_levels_exist(self):
        """Test all risk levels are defined."""
        assert LicenseRisk.PERMISSIVE.value == "permissive"
        assert LicenseRisk.CONDITIONAL.value == "conditional"
        assert LicenseRisk.NON_COMMERCIAL.value == "non_commercial"
        assert LicenseRisk.RESTRICTED.value == "restricted"
        assert LicenseRisk.UNKNOWN.value == "unknown"


class TestLicenseClassification:
    """Tests for LICENSE_CLASSIFICATION dict."""

    def test_permissive_licenses(self):
        """Test permissive licenses are classified correctly."""
        permissive = ["apache-2.0", "mit", "bsd-3-clause", "cc0-1.0", "unlicense"]
        for license_id in permissive:
            assert LICENSE_CLASSIFICATION[license_id] == LicenseRisk.PERMISSIVE

    def test_conditional_licenses(self):
        """Test conditional licenses are classified correctly."""
        conditional = ["llama2", "llama3", "gemma", "openrail"]
        for license_id in conditional:
            assert LICENSE_CLASSIFICATION[license_id] == LicenseRisk.CONDITIONAL

    def test_non_commercial_licenses(self):
        """Test non-commercial licenses are classified correctly."""
        non_commercial = ["cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0"]
        for license_id in non_commercial:
            assert LICENSE_CLASSIFICATION[license_id] == LicenseRisk.NON_COMMERCIAL


class TestLicenseRestrictions:
    """Tests for LICENSE_RESTRICTIONS dict."""

    def test_llama_restrictions(self):
        """Test Llama license restrictions."""
        for llama_version in ["llama2", "llama3", "llama3.1", "llama3.2", "llama3.3"]:
            restrictions = LICENSE_RESTRICTIONS[llama_version]
            assert "commercial-use-up-to-700M-MAU" in restrictions
            assert "attribution-required: 'Built with Llama'" in restrictions

    def test_cc_nc_restrictions(self):
        """Test CC-BY-NC restrictions."""
        restrictions = LICENSE_RESTRICTIONS["cc-by-nc-4.0"]
        assert "non-commercial-only" in restrictions
        assert "attribution-required" in restrictions


class TestCheckModelLicense:
    """Tests for check_model_license function."""

    def test_apache_license(self):
        """Test Apache 2.0 license detection."""
        with patch("hfl.hub.license_checker.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.card_data = MagicMock()
            mock_info.card_data.license = "apache-2.0"
            mock_info.card_data.license_name = None
            mock_info.card_data.license_link = None
            mock_info.gated = False
            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = check_model_license("test/model")

            assert result.license_id == "apache-2.0"
            assert result.risk == LicenseRisk.PERMISSIVE
            assert result.gated is False

    def test_llama_license(self):
        """Test Llama license detection."""
        with patch("hfl.hub.license_checker.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.card_data = MagicMock()
            mock_info.card_data.license = "llama3.1"
            mock_info.card_data.license_name = None
            mock_info.card_data.license_link = None
            mock_info.gated = True
            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = check_model_license("meta-llama/Llama-3.1-8B")

            assert result.risk == LicenseRisk.CONDITIONAL
            assert result.gated is True
            assert len(result.restrictions) > 0

    def test_license_from_tags_fallback(self):
        """Test license detection from tags when card_data is missing."""
        with patch("hfl.hub.license_checker.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.card_data = None
            mock_info.tags = ["license:mit", "text-generation"]
            mock_info.gated = False
            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = check_model_license("test/model")

            assert result.license_id == "mit"
            assert result.risk == LicenseRisk.PERMISSIVE

    def test_unknown_license(self):
        """Test unknown license detection."""
        with patch("hfl.hub.license_checker.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.card_data = MagicMock()
            mock_info.card_data.license = "some-custom-license"
            mock_info.card_data.license_name = None
            mock_info.card_data.license_link = None
            mock_info.gated = False
            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = check_model_license("test/model")

            assert result.risk == LicenseRisk.UNKNOWN

    def test_license_name_used_for_other(self):
        """Test that license_name is used when license is 'other'."""
        with patch("hfl.hub.license_checker.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.card_data = MagicMock()
            mock_info.card_data.license = "other"
            mock_info.card_data.license_name = "qwen"
            mock_info.card_data.license_link = "https://example.com/license"
            mock_info.gated = False
            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = check_model_license("Qwen/Qwen2-7B")

            assert result.license_id == "qwen"
            assert result.risk == LicenseRisk.CONDITIONAL

    def test_license_url_default(self):
        """Test that default license URL is generated."""
        with patch("hfl.hub.license_checker.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.card_data = MagicMock()
            mock_info.card_data.license = "mit"
            mock_info.card_data.license_name = None
            mock_info.card_data.license_link = None
            mock_info.gated = False
            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = check_model_license("test/model")

            assert result.url == "https://huggingface.co/test/model#license"

    def test_partial_match_classification(self):
        """Test partial matching for license names."""
        with patch("hfl.hub.license_checker.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.card_data = MagicMock()
            mock_info.card_data.license = "other"
            mock_info.card_data.license_name = "llama3.1-community"
            mock_info.card_data.license_link = None
            mock_info.gated = False
            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = check_model_license("test/model")

            # Should match llama3.1 via partial match
            assert result.risk == LicenseRisk.CONDITIONAL


class TestRequireUserAcceptance:
    """Tests for require_user_acceptance function."""

    def test_permissive_license_auto_accepts(self):
        """Test that permissive licenses are auto-accepted."""
        license_info = LicenseInfo(
            license_id="apache-2.0",
            license_name="Apache 2.0",
            risk=LicenseRisk.PERMISSIVE,
            restrictions=[],
            url="https://example.com",
            gated=False,
        )

        with patch("rich.console.Console"):
            result = require_user_acceptance(license_info, "test/model")

        assert result is True

    def test_non_commercial_requires_confirmation(self):
        """Test that non-commercial licenses require user confirmation."""
        license_info = LicenseInfo(
            license_id="cc-by-nc-4.0",
            license_name="CC BY-NC 4.0",
            risk=LicenseRisk.NON_COMMERCIAL,
            restrictions=["non-commercial-only"],
            url="https://example.com",
            gated=False,
        )

        with patch("rich.console.Console"):
            with patch("rich.panel.Panel"):
                with patch("typer.confirm") as mock_confirm:
                    mock_confirm.return_value = True
                    result = require_user_acceptance(license_info, "test/model")

        assert result is True
        mock_confirm.assert_called_once()

    def test_user_rejects_license(self):
        """Test that user can reject a license."""
        license_info = LicenseInfo(
            license_id="cc-by-nc-4.0",
            license_name="CC BY-NC 4.0",
            risk=LicenseRisk.NON_COMMERCIAL,
            restrictions=["non-commercial-only"],
            url="https://example.com",
            gated=False,
        )

        with patch("rich.console.Console"):
            with patch("rich.panel.Panel"):
                with patch("typer.confirm") as mock_confirm:
                    mock_confirm.return_value = False
                    result = require_user_acceptance(license_info, "test/model")

        assert result is False

    def test_restricted_license_panel(self):
        """Test that restricted licenses show correct panel."""
        license_info = LicenseInfo(
            license_id="restricted-license",
            license_name="Restricted",
            risk=LicenseRisk.RESTRICTED,
            restrictions=["research-only"],
            url="https://example.com",
            gated=False,
        )

        with patch("rich.console.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            with patch("rich.panel.Panel"):
                with patch("typer.confirm") as mock_confirm:
                    mock_confirm.return_value = True
                    require_user_acceptance(license_info, "test/model")

                    # Verify panel was created with correct title
                    mock_console.print.assert_called()

    def test_unknown_license_shows_warning(self):
        """Test that unknown licenses show appropriate warning."""
        license_info = LicenseInfo(
            license_id="unknown-license",
            license_name="Unknown",
            risk=LicenseRisk.UNKNOWN,
            restrictions=[],
            url="https://example.com",
            gated=False,
        )

        with patch("rich.console.Console"):
            with patch("rich.panel.Panel"):
                with patch("typer.confirm") as mock_confirm:
                    mock_confirm.return_value = True
                    result = require_user_acceptance(license_info, "test/model")

        assert result is True

    def test_gated_model_shows_gated_text(self):
        """Test that gated models show appropriate text."""
        license_info = LicenseInfo(
            license_id="llama3.1",
            license_name="Llama 3.1",
            risk=LicenseRisk.CONDITIONAL,
            restrictions=["attribution-required"],
            url="https://example.com",
            gated=True,
        )

        with patch("rich.console.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            with patch("rich.panel.Panel"):
                with patch("typer.confirm") as mock_confirm:
                    mock_confirm.return_value = True
                    require_user_acceptance(license_info, "test/model")


class TestGetLicenseSummary:
    """Tests for get_license_summary function."""

    def test_permissive_summary(self):
        """Test summary for permissive license."""
        license_info = LicenseInfo(
            license_id="apache-2.0",
            license_name="Apache 2.0",
            risk=LicenseRisk.PERMISSIVE,
            restrictions=[],
            url="https://example.com",
            gated=False,
        )

        result = get_license_summary(license_info)

        assert "apache-2.0" in result
        assert "[green]OK[/]" in result

    def test_conditional_summary(self):
        """Test summary for conditional license."""
        license_info = LicenseInfo(
            license_id="llama3.1",
            license_name="Llama 3.1",
            risk=LicenseRisk.CONDITIONAL,
            restrictions=[],
            url="https://example.com",
            gated=False,
        )

        result = get_license_summary(license_info)

        assert "llama3.1" in result
        assert "[yellow]![/]" in result

    def test_non_commercial_summary(self):
        """Test summary for non-commercial license."""
        license_info = LicenseInfo(
            license_id="cc-by-nc-4.0",
            license_name="CC BY-NC 4.0",
            risk=LicenseRisk.NON_COMMERCIAL,
            restrictions=[],
            url="https://example.com",
            gated=False,
        )

        result = get_license_summary(license_info)

        assert "cc-by-nc-4.0" in result
        assert "[red]NC[/]" in result

    def test_restricted_summary(self):
        """Test summary for restricted license."""
        license_info = LicenseInfo(
            license_id="restricted",
            license_name="Restricted",
            risk=LicenseRisk.RESTRICTED,
            restrictions=[],
            url="https://example.com",
            gated=False,
        )

        result = get_license_summary(license_info)

        assert "[red]R[/]" in result

    def test_unknown_summary(self):
        """Test summary for unknown license."""
        license_info = LicenseInfo(
            license_id="custom",
            license_name="Custom",
            risk=LicenseRisk.UNKNOWN,
            restrictions=[],
            url="https://example.com",
            gated=False,
        )

        result = get_license_summary(license_info)

        assert "[yellow]?[/]" in result
