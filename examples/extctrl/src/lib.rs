mod messages;
mod southbound;

pub use messages::{LinkStateEntry, LinkStateMsg, PathInstructions};
pub use southbound::{PathAction, Southbound};

/// Convert seconds to time slots at given accuracy.
pub fn sec_to_time_slot(sec: f64, accuracy: u64) -> u64 {
    (sec * accuracy as f64).round() as u64
}
