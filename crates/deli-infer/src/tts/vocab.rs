use std::collections::HashMap;

pub(crate) fn vocab() -> HashMap<char, i64> {
    HashMap::from([
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
        ('\u{201c}', 14),
        ('\u{201d}', 15),
        (' ', 16),
        ('\u{0303}', 17),
        ('ʣ', 18),
        ('ʥ', 19),
        ('ʦ', 20),
        ('ʨ', 21),
        ('ᵝ', 22),
        ('ŋ', 23),
        ('ɐ', 24),
        ('ɑ', 25),
        ('ɒ', 26),
        ('ɓ', 27),
        ('ɔ', 28),
        ('ɕ', 29),
        ('ɖ', 30),
        ('ɗ', 31),
        ('ə', 32),
        ('ɚ', 33),
        ('ɛ', 34),
        ('ɜ', 35),
        ('ɞ', 36),
        ('ɟ', 37),
        ('ɠ', 38),
        ('ɡ', 39),
        ('ɢ', 40),
        ('ɣ', 41),
        ('ɤ', 42),
        ('a', 43),
        ('ɦ', 44),
        ('ɧ', 45),
        ('ɨ', 46),
        ('ɪ', 47),
        ('ɫ', 48),
        ('ɬ', 49),
        ('ɭ', 50),
        ('ɮ', 51),
        ('ɯ', 52),
        ('ɰ', 53),
        ('ɱ', 54),
        ('ɲ', 55),
        ('ɳ', 56),
        ('ɴ', 57),
        ('ɵ', 58),
        ('ɶ', 59),
        ('ɸ', 60),
        ('ɹ', 61),
        ('ɺ', 62),
        ('ɻ', 63),
        ('ɽ', 64),
        ('ɾ', 65),
        ('ʀ', 66),
        ('ʁ', 67),
        ('ʂ', 68),
        ('ʃ', 69),
        ('ʄ', 70),
        ('ʈ', 71),
        ('ʉ', 72),
        ('ʊ', 73),
        ('ʋ', 74),
        ('ʌ', 75),
        ('ʍ', 76),
        ('ʎ', 77),
        ('ʏ', 78),
        ('ʐ', 79),
        ('ʑ', 80),
        ('ʒ', 81),
        ('ʓ', 82),
        ('ʔ', 83),
        ('ʕ', 84),
        ('ʘ', 85),
        ('ʙ', 86),
        ('ʛ', 87),
        ('ʜ', 88),
        ('ʝ', 89),
        ('ʟ', 90),
        ('ʡ', 91),
        ('ʢ', 92),
        ('ˈ', 156),
        ('ˌ', 157),
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
        assert_eq!(vocab().len(), 92);
    }

    #[test]
    fn test_vocab_key_mappings() {
        let v = vocab();
        assert_eq!(v.get(&'a'), Some(&43));
        assert_eq!(v.get(&' '), Some(&16));
        assert_eq!(v.get(&'ɪ'), Some(&47));
        assert_eq!(v.get(&'ˈ'), Some(&156));
        assert_eq!(v.get(&'.'), Some(&4));
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
        let tokens = tokenize("XYZ123", &v);
        assert!(tokens.is_empty());
    }
}
