import gzip
import os
import shutil
import tarfile
import zipfile
from typing import Any, Dict

from app.services import logger
from app.services.temp_directory_analyzer import TemporaryDirectoryAnalyzer


class CompressedSourceAnalyzer(TemporaryDirectoryAnalyzer):
    """Analyzer for compressed source packages (.zip, .tar.gz, etc.)."""

    def _is_tar_file(self) -> bool:
        """Check if the file is a tar archive."""
        return (self.package_path.endswith('.tar') or
                self.package_path.endswith('.tar.gz') or
                self.package_path.endswith('.tgz'))

    def _is_zip_file(self) -> bool:
        """Check if the file is a zip archive."""
        return self.package_path.endswith('.zip')

    def _is_gzip_file(self) -> bool:
        """Check if the file is a gzip archive."""
        return self.package_path.endswith('.gz') and not self._is_tar_file()

    def extract_contents(self):
        """Extract contents based on the compression type."""
        logger.info(f"Extracting compressed file: {self.package_path}")

        if self._is_zip_file():
            self._extract_zip()
        elif self._is_tar_file():
            self._extract_tar()
        elif self._is_gzip_file():
            self._extract_gzip()
        else:
            raise ValueError(f"Unsupported compression format: {self.package_path}")

    def _extract_zip(self):
        """Extract a ZIP archive."""
        with zipfile.ZipFile(self.package_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)

    def _extract_tar(self):
        """Extract a TAR archive (including .tar.gz)."""
        with tarfile.open(self.package_path, 'r:*') as tar_ref:
            # Check for potential path traversal vulnerabilities
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted path traversal in tar file")
                tar.extractall(path)

            safe_extract(tar_ref, self.temp_dir)

    def _extract_gzip(self):
        """Extract a GZIP file."""
        output_path = os.path.join(self.temp_dir,
                                   os.path.basename(self.package_path)[:-3])
        with gzip.open(self.package_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def generate_report(self) -> str:
        """Generate a report including compression-specific information."""
        # Store the original package path
        original_path = self.package_path

        try:
            # Temporarily set package path to the extracted contents for correct relative paths
            if self.temp_dir:
                package_root = self._find_package_root()
                if package_root:
                    self.package_path = package_root

            # Get base report with correct paths
            report = super().generate_report()

            # Add compression information
            compression_info = self._get_compression_info()
            compression_section = [
                "\nCompressed Package Information",
                "===========================",
                f"Compression Type: {compression_info['type']}",
                f"Original File: {original_path}",
                f"Original Size: {compression_info['original_size']}",
                f"Compressed Size: {compression_info['compressed_size']}",
                f"Compression Ratio: {compression_info['ratio']:.2f}%"
            ]

            return report + '\n\n' + '\n'.join(compression_section)

        finally:
            # Restore the original package path
            self.package_path = original_path

    def _get_compression_info(self) -> Dict[str, Any]:
        """Get information about the compression."""
        original_size = 0
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                original_size += os.path.getsize(file_path)

        compressed_size = os.path.getsize(self.original_path)

        compression_type = 'Unknown'
        if self._is_zip_file():
            compression_type = 'ZIP'
        elif self._is_tar_file():
            compression_type = 'TAR' + ('.GZ' if self.package_path.endswith('.gz') else '')
        elif self._is_gzip_file():
            compression_type = 'GZIP'

        return {
            'type': compression_type,
            'original_size': self._format_size(original_size),
            'compressed_size': self._format_size(compressed_size),
            'ratio': (compressed_size / original_size * 100) if original_size > 0 else 0
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"
