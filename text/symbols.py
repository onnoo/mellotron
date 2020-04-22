""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

_punctuation = '!\'",.:;? '
_math = '#%&*+-/[]()'
_special = '_@©°½—₩€$'
_accented = 'áçéêëñöøćž'
_numbers = '0123456789'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_capitals = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

JAMO_LEADS = [chr(_) for _ in range(0x1100, 0x1113)]
JAMO_LEADS.remove(JAMO_LEADS[11])
JAMO_VOWELS = [chr(_) for _ in range(0x1161, 0x1176)]
JAMO_TAILS = [chr(_) for _ in range(0x11A8, 0x11C3)]
JAMO_TAILS = [JAMO_TAILS[0], JAMO_TAILS[3], JAMO_TAILS[6], JAMO_TAILS[7], JAMO_TAILS[15], JAMO_TAILS[16], JAMO_TAILS[20]]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as
# uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
en_symbols = list(_punctuation + _math + _special + _accented + _numbers + _letters) + _arpabet
symbols = list(_punctuation + _math + _special + _accented + _numbers + _capitals) + JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + _arpabet[:-20]
