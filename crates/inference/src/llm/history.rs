use tokio::sync::RwLock;

#[derive(Clone, PartialEq)]
pub enum Speaker {
    User,
    Model,
}

#[derive(Clone)]
pub struct Entry {
    pub timestamp: u64,
    pub speaker: Speaker,
    pub sentence: String,
}

pub struct History {
    pub entries: RwLock<Vec<Entry>>,
}

impl History {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }

    pub async fn add_entry(&self, entry: Entry) {
        self.entries.write().await.push(entry);
    }

    pub async fn get_most_recent(&self, count: usize) -> Vec<Entry> {
        let mut entries = Vec::<Entry>::new();
        let read = self.entries.read().await;
        let len = read.len();
        let mut max = len;
        if max > count {
            max = count;
        }
        for i in 0..max {
            entries.push(read[len - max + i].clone());
        }
        entries
    }

    pub async fn clear(&self) {
        self.entries.write().await.clear();
    }
}
