#
# Windows OS wav file player
#
# Some parts Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
# Some parts Inspired by (but not copied from) Michael Gundlach <gundlach@gmail.com>'s mp3play:
#         https://github.com/michaelgundlach/mp3play
# Some parts inpsired by  playsound pip package
# Some parts written by me
class SoundPlayer:
    def __init__(self):
        pass

    def playWav(self, data, rate, normalize=True):
        import wave

        scaled, nchan = self.__validate_and_normalize_with_numpy(data, normalize)
        #waveobj = self.__make_wav(data, rate, normalize)
        waveFile = wave.open("temp.wav", "wb")
        waveFile.setnchannels(nchan)
        waveFile.setframerate(rate)
        waveFile.setsampwidth(2)
        waveFile.setcomptype('NONE', 'NONE')
        waveFile.writeframes(scaled)
        waveFile.close()
        self._playsoundWin("temp.wav")

    def __validate_and_normalize_with_numpy(self, data, normalize):
        import numpy as np

        data = np.array(data, dtype=float)
        if len(data.shape) == 1:
            nchan = 1
        elif len(data.shape) == 2:
            # In wave files,channels are interleaved. E.g.,
            # "L1R1L2R2..." for stereo. See
            # http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
            # for channel ordering
            nchan = data.shape[0]
            data = data.T.ravel()
        else:
            raise ValueError('Array audio input must be a 1D or 2D array')

        max_abs_value = np.max(np.abs(data))
        normalization_factor = self.__get_normalization_factor(max_abs_value, normalize)
        scaled = data / normalization_factor * 32767
        return scaled.astype('<h').tostring(), nchan

    def __get_normalization_factor(self, max_abs_value, normalize):
        if not normalize and max_abs_value > 1:
            raise ValueError('Audio data must be between -1 and 1 when normalize=False.')
        return max_abs_value if normalize else 1

    def _playsoundWin(self, sound, block=True):
        '''
        Utilizes windll.winmm. Tested and known to work with MP3 and WAVE on
        Windows 7 with Python 2.7. Probably works with more file formats.
        Probably works on Windows XP thru Windows 10. Probably works with all
        versions of Python.

        Inspired by (but not copied from) Michael Gundlach <gundlach@gmail.com>'s mp3play:
        https://github.com/michaelgundlach/mp3play

        I never would have tried using windll.winmm without seeing his code.
        '''
        from ctypes import c_buffer, windll
        from random import random
        from time import sleep
        from sys import getfilesystemencoding

        def winCommand(*command):
            buf = c_buffer(255)
            command = ' '.join(command).encode(getfilesystemencoding())
            errorCode = int(windll.winmm.mciSendStringA(command, buf, 254, 0))
            if errorCode:
                errorBuffer = c_buffer(255)
                windll.winmm.mciGetErrorStringA(errorCode, errorBuffer, 254)
                exceptionMessage = ('\n    Error ' + str(errorCode) + ' for command:'
                                                                      '\n        ' + command.decode() +
                                    '\n    ' + errorBuffer.value.decode())
                raise PlaysoundException(exceptionMessage)
            return buf.value

        alias = 'playsound_' + str(random())
        winCommand('open "' + sound + '" alias', alias)
        winCommand('set', alias, 'time format milliseconds')
        durationInMS = winCommand('status', alias, 'length')
        winCommand('play', alias, 'from 0 to', durationInMS.decode())

        if block:
            sleep(float(durationInMS) / 1000.0)
            winCommand('close', alias)

class PlaysoundException(Exception):
    pass