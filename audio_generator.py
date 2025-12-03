"""
Audio Generation Module
Converts text summaries and answers to speech audio files.
"""
import os
import io
from typing import Optional, Union
from pathlib import Path

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

from config import Config


class AudioGenerator:
    """Generates audio from text using various TTS engines."""
    
    def __init__(self, engine: str = "gtts"):
        """
        Initialize audio generator.
        
        Args:
            engine: TTS engine to use - 'gtts' (Google TTS, requires internet) 
                    or 'pyttsx3' (offline, system voices)
        """
        self.engine = engine
        self._check_availability()
    
    def _check_availability(self):
        """Check which TTS engines are available."""
        if self.engine == "gtts" and not GTTS_AVAILABLE:
            if PYTTSX3_AVAILABLE:
                print("Warning: gTTS not available, falling back to pyttsx3")
                self.engine = "pyttsx3"
            else:
                raise ImportError("No TTS engine available. Install gtts or pyttsx3.")
        elif self.engine == "pyttsx3" and not PYTTSX3_AVAILABLE:
            if GTTS_AVAILABLE:
                print("Warning: pyttsx3 not available, falling back to gTTS")
                self.engine = "gtts"
            else:
                raise ImportError("No TTS engine available. Install gtts or pyttsx3.")
    
    def generate_audio(
        self,
        text: str,
        output_path: Optional[str] = None,
        language: str = "en",
        slow: bool = False
    ) -> Union[bytes, str]:
        """
        Generate audio from text.
        
        Args:
            text: Text to convert to speech
            output_path: Optional path to save audio file. If None, returns bytes
            language: Language code (default: 'en' for English)
            slow: Whether to speak slowly (gTTS only)
        
        Returns:
            If output_path is None, returns audio bytes. Otherwise returns file path.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Limit text length to avoid issues
        max_length = 5000  # Characters
        if len(text) > max_length:
            text = text[:max_length] + "... [text truncated for audio]"
        
        if self.engine == "gtts":
            return self._generate_gtts(text, output_path, language, slow)
        elif self.engine == "pyttsx3":
            return self._generate_pyttsx3(text, output_path, language)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
    
    def _generate_gtts(
        self,
        text: str,
        output_path: Optional[str],
        language: str,
        slow: bool
    ) -> Union[bytes, str]:
        """Generate audio using Google Text-to-Speech."""
        if not GTTS_AVAILABLE:
            raise ImportError("gTTS not available. Install with: pip install gtts")
        
        try:
            tts = gTTS(text=text, lang=language, slow=slow)
            
            if output_path:
                tts.save(output_path)
                return output_path
            else:
                # Save to bytes buffer
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                return audio_buffer.getvalue()
        except Exception as e:
            raise RuntimeError(f"Error generating audio with gTTS: {str(e)}")
    
    def _generate_pyttsx3(
        self,
        text: str,
        output_path: Optional[str],
        language: str
    ) -> Union[bytes, str]:
        """Generate audio using pyttsx3 (offline TTS)."""
        if not PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 not available. Install with: pip install pyttsx3")
        
        try:
            engine = pyttsx3.init()
            
            # Set properties (optional)
            # You can set voice, rate, volume here
            # voices = engine.getProperty('voices')
            # engine.setProperty('voice', voices[0].id)  # Use first available voice
            # engine.setProperty('rate', 150)  # Speed of speech
            
            if output_path:
                engine.save_to_file(text, output_path)
                engine.runAndWait()
                return output_path
            else:
                # For pyttsx3, we need to save to a temp file first
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    temp_path = tmp_file.name
                
                engine.save_to_file(text, temp_path)
                engine.runAndWait()
                
                # Read the file and return bytes
                with open(temp_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Clean up temp file
                os.unlink(temp_path)
                return audio_bytes
        except Exception as e:
            raise RuntimeError(f"Error generating audio with pyttsx3: {str(e)}")
    
    def generate_audio_file(
        self,
        text: str,
        filename: str,
        output_dir: str = "./audio_output",
        language: str = "en"
    ) -> str:
        """
        Generate audio file and save it.
        
        Args:
            text: Text to convert to speech
            filename: Name of the output file (without extension)
            output_dir: Directory to save the file
            language: Language code
        
        Returns:
            Full path to the generated audio file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine file extension based on engine
        if self.engine == "gtts":
            extension = ".mp3"
        else:
            extension = ".wav"
        
        output_path = os.path.join(output_dir, f"{filename}{extension}")
        
        self.generate_audio(text, output_path, language)
        return output_path


def generate_audio_from_text(
    text: str,
    output_path: Optional[str] = None,
    engine: str = "gtts",
    language: str = "en"
) -> Union[bytes, str]:
    """
    Convenience function to generate audio from text.
    
    Args:
        text: Text to convert to speech
        output_path: Optional path to save audio file
        engine: TTS engine ('gtts' or 'pyttsx3')
        language: Language code
    
    Returns:
        Audio bytes or file path
    """
    generator = AudioGenerator(engine=engine)
    return generator.generate_audio(text, output_path, language)

