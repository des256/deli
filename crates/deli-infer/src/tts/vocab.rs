use std::collections::HashMap;

pub(crate) fn vocab() -> HashMap<char, i64> {
    HashMap::from([
        // Punctuation
        (';', 1),
        (':', 2),
        (',', 3),
        ('.', 4),
        ('!', 5),
        ('?', 6),
        ('—', 9),
        ('…', 10),
        ('"', 11),
        ('(', 12),
        (')', 13),
        ('\u{201c}', 14), // "
        ('\u{201d}', 15), // "
        (' ', 16),
        ('\u{0303}', 17), // combining tilde
        ('ʣ', 18),
        ('ʥ', 19),
        ('ʦ', 20),
        ('ʨ', 21),
        ('ᵝ', 22),
        ('\u{AB67}', 23),
        // Uppercase letters (sparse)
        ('A', 24),
        ('I', 25),
        ('O', 31),
        ('Q', 33),
        ('S', 35),
        ('T', 36),
        ('W', 39),
        ('Y', 41),
        ('ᵊ', 42),
        // Lowercase letters
        ('a', 43),
        ('b', 44),
        ('c', 45),
        ('d', 46),
        ('e', 47),
        ('f', 48),
        ('h', 50),
        ('i', 51),
        ('j', 52),
        ('k', 53),
        ('l', 54),
        ('m', 55),
        ('n', 56),
        ('o', 57),
        ('p', 58),
        ('q', 59),
        ('r', 60),
        ('s', 61),
        ('t', 62),
        ('u', 63),
        ('v', 64),
        ('w', 65),
        ('x', 66),
        ('y', 67),
        ('z', 68),
        // IPA vowels and consonants
        ('ɑ', 69),
        ('ɐ', 70),
        ('ɒ', 71),
        ('æ', 72),
        ('β', 75),
        ('ɔ', 76),
        ('ɕ', 77),
        ('ç', 78),
        ('ɖ', 80),
        ('ð', 81),
        ('ʤ', 82),
        ('ə', 83),
        ('ɚ', 85),
        ('ɛ', 86),
        ('ɜ', 87),
        ('ɟ', 90),
        ('ɡ', 92),
        ('ɥ', 99),
        ('ɨ', 101),
        ('ɪ', 102),
        ('ʝ', 103),
        ('ɯ', 110),
        ('ɰ', 111),
        ('ŋ', 112),
        ('ɳ', 113),
        ('ɲ', 114),
        ('ɴ', 115),
        ('ø', 116),
        ('ɸ', 118),
        ('θ', 119),
        ('œ', 120),
        ('ɹ', 123),
        ('ɾ', 125),
        ('ɻ', 126),
        ('ʁ', 128),
        ('ɽ', 129),
        ('ʂ', 130),
        ('ʃ', 131),
        ('ʈ', 132),
        ('ʧ', 133),
        ('ʊ', 135),
        ('ʋ', 136),
        ('ʌ', 138),
        ('ɣ', 139),
        ('ɤ', 140),
        ('χ', 142),
        ('ʎ', 143),
        ('ʒ', 147),
        ('ʔ', 148),
        // Prosodic markers
        ('ˈ', 156),
        ('ˌ', 157),
        ('ː', 158),
        ('ʰ', 162),
        ('ʲ', 164),
        // Intonation arrows
        ('↓', 169),
        ('→', 171),
        ('↗', 172),
        ('↘', 173),
        ('ᵻ', 177),
    ])
}

pub(crate) fn tokenize(phonemes: &str, vocab: &HashMap<char, i64>) -> Vec<i64> {
    phonemes
        .chars()
        .filter_map(|c| vocab.get(&c).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size() {
        assert_eq!(vocab().len(), 114);
    }

    #[test]
    fn test_vocab_key_mappings() {
        let v = vocab();
        assert_eq!(v.get(&'a'), Some(&43));
        assert_eq!(v.get(&' '), Some(&16));
        assert_eq!(v.get(&'ɪ'), Some(&102));
        assert_eq!(v.get(&'ˈ'), Some(&156));
        assert_eq!(v.get(&'.'), Some(&4));
        assert_eq!(v.get(&'ŋ'), Some(&112));
        assert_eq!(v.get(&'ː'), Some(&158));
        assert_eq!(v.get(&'θ'), Some(&119));
        assert_eq!(v.get(&'ð'), Some(&81));
    }

    #[test]
    fn test_tokenize_known_chars() {
        let v = vocab();
        // "a " should produce [43, 16]
        let tokens = tokenize("a ", &v);
        assert_eq!(tokens, vec![43, 16]);
    }

    #[test]
    fn test_tokenize_unknown_chars_skipped() {
        let v = vocab();
        // 'X' and 'Z' are not in the vocab, should be silently skipped
        let tokens = tokenize("XaZ ", &v);
        assert_eq!(tokens, vec![43, 16]);
    }

    #[test]
    fn test_tokenize_empty() {
        let v = vocab();
        let tokens = tokenize("", &v);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_all_unknown() {
        let v = vocab();
        let tokens = tokenize("123", &v);
        assert!(tokens.is_empty());
    }
}
