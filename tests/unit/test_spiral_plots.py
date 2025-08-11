"""Unit tests for spiral plotting functionality."""

import pathlib
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from graphomotor.core import config
from graphomotor.plot import spiral_plots
from graphomotor.io import reader


class TestSpiralPlots:
    """Test class for spiral plotting functionality."""
    
    @pytest.fixture
    def sample_spiral_file(self):
        """Path to a sample spiral CSV file."""
        return pathlib.Path(__file__).parent.parent / "sample_data" / "[5123456]65318bf53c36ce79135b1049-648c7d0e8819c1120b4f708d-spiral_trace1_Dom.csv"
    
    @pytest.fixture
    def sample_spiral_data(self, sample_spiral_file):
        """Load sample spiral data."""
        return reader.load_spiral(sample_spiral_file)
    
    def test_plot_single_spiral_basic(self, sample_spiral_file):
        """Test basic single spiral plotting functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "test_plot.png"
            
            # Should not raise any exceptions
            spiral_plots.plot_single_spiral(
                input_path=sample_spiral_file,
                output_path=output_path
            )
            
            # Check that file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_plot_single_spiral_with_reference(self, sample_spiral_file):
        """Test single spiral plotting with reference overlay."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "test_plot_ref.png"
            spiral_config = config.SpiralConfig()
            
            spiral_plots.plot_single_spiral(
                input_path=sample_spiral_file,
                output_path=output_path,
                include_reference=True,
                spiral_config=spiral_config
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_plot_single_spiral_with_color_segments(self, sample_spiral_file):
        """Test single spiral plotting with colored segments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "test_plot_colored.png"
            
            spiral_plots.plot_single_spiral(
                input_path=sample_spiral_file,
                output_path=output_path,
                color_by_segment=True
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_plot_single_spiral_auto_output(self, sample_spiral_file):
        """Test single spiral plotting with auto-generated output path."""
        # Should not raise any exceptions
        spiral_plots.plot_single_spiral(input_path=sample_spiral_file)
        
        # Check that default output file was created
        expected_output = sample_spiral_file.parent / f"{sample_spiral_file.stem}_plot.png"
        assert expected_output.exists()
        
        # Clean up
        expected_output.unlink()
    
    def test_plot_spiral_batch_individual(self):
        """Test batch plotting with individual plots."""
        sample_dir = pathlib.Path(__file__).parent.parent / "sample_data"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            
            spiral_plots.plot_spiral_batch(
                input_path=sample_dir,
                output_path=output_dir,
                group_by=None  # Individual plots
            )
            
            # Should create individual plot files
            plot_files = list(output_dir.glob("*.png"))
            assert len(plot_files) >= 1  # At least one plot should be created
            
            for plot_file in plot_files:
                assert plot_file.stat().st_size > 0
    
    def test_plot_spiral_batch_grouped_by_task(self):
        """Test batch plotting with grouping by task."""
        sample_dir = pathlib.Path(__file__).parent.parent / "sample_data"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            
            spiral_plots.plot_spiral_batch(
                input_path=sample_dir,
                output_path=output_dir,
                group_by="task"
            )
            
            # Should create grouped plot files
            plot_files = list(output_dir.glob("*.png"))
            assert len(plot_files) >= 1
            
            # Check that at least one file contains task grouping info
            task_group_files = [f for f in plot_files if "spirals_task_" in f.name]
            assert len(task_group_files) >= 1
    
    def test_plot_spiral_batch_invalid_group_by(self):
        """Test batch plotting with invalid group_by parameter."""
        sample_dir = pathlib.Path(__file__).parent.parent / "sample_data"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            
            with pytest.raises(ValueError, match="group_by must be one of"):
                spiral_plots.plot_spiral_batch(
                    input_path=sample_dir,
                    output_path=output_dir,
                    group_by="invalid_option"
                )
    
    def test_plot_spiral_batch_nonexistent_directory(self):
        """Test batch plotting with non-existent input directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = pathlib.Path(temp_dir) / "nonexistent"
            output_dir = pathlib.Path(temp_dir) / "output"
            
            with pytest.raises(IOError, match="Input path is not a directory"):
                spiral_plots.plot_spiral_batch(
                    input_path=nonexistent_dir,
                    output_path=output_dir
                )
    
    def test_plot_spiral_batch_no_csv_files(self):
        """Test batch plotting with directory containing no CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = pathlib.Path(temp_dir) / "empty"
            empty_dir.mkdir()
            output_dir = pathlib.Path(temp_dir) / "output"
            
            with pytest.raises(IOError, match="No CSV files found in directory"):
                spiral_plots.plot_spiral_batch(
                    input_path=empty_dir,
                    output_path=output_dir
                )
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_single_spiral_matplotlib_calls(self, mock_close, mock_savefig, sample_spiral_file):
        """Test that matplotlib functions are called correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "test_plot.png"
            
            spiral_plots.plot_single_spiral(
                input_path=sample_spiral_file,
                output_path=output_path
            )
            
            # Check that savefig was called with correct parameters
            mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
            mock_close.assert_called_once()
    
    def test_plot_spiral_on_axis_helper(self, sample_spiral_data):
        """Test the _plot_spiral_on_axis helper function."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            spiral_plots._plot_spiral_on_axis(
                ax=mock_ax,
                spiral=sample_spiral_data,
                include_reference=False,
                color_by_segment=False,
                spiral_config=None
            )
            
            # Check that basic axis methods were called
            mock_ax.plot.assert_called()
            mock_ax.set_aspect.assert_called_with('equal', adjustable='box')
            mock_ax.set_title.assert_called()
            mock_ax.grid.assert_called_with(True, alpha=0.3)