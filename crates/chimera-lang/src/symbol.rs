//! Interned identifiers.
//!
//! Every name in a Chimera program (variables, fields, methods, keywords used
//! as names) is interned into a [`Sym`], a 4-byte copyable handle. Comparing
//! two symbols is an integer comparison, which keeps environment lookups and
//! attribute access cheap.
//!
//! The interner is thread-local: an engine, the programs it parses and the
//! values it creates all live on one thread.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// An interned string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sym(u32);

#[derive(Default)]
struct Interner {
    map: HashMap<Rc<str>, Sym>,
    names: Vec<Rc<str>>,
}

thread_local! {
    static INTERNER: RefCell<Interner> = RefCell::new(Interner::default());
}

impl Sym {
    /// Interns `name`, returning the same symbol for equal strings.
    pub fn new(name: &str) -> Sym {
        INTERNER.with(|cell| {
            let mut interner = cell.borrow_mut();
            if let Some(&sym) = interner.map.get(name) {
                return sym;
            }
            let rc: Rc<str> = Rc::from(name);
            let sym = Sym(interner.names.len() as u32);
            interner.names.push(rc.clone());
            interner.map.insert(rc, sym);
            sym
        })
    }

    /// The symbol's text.
    pub fn as_str(self) -> Rc<str> {
        INTERNER.with(|cell| cell.borrow().names[self.0 as usize].clone())
    }

    /// Whether the symbol's text equals `s`.
    pub fn is(self, s: &str) -> bool {
        INTERNER.with(|cell| &*cell.borrow().names[self.0 as usize] == s)
    }
}

impl fmt::Debug for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl fmt::Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

impl From<&str> for Sym {
    fn from(s: &str) -> Self {
        Sym::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_stable() {
        let a = Sym::new("chimera");
        let b = Sym::new("chimera");
        let c = Sym::new("hydra");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(&*a.as_str(), "chimera");
        assert!(c.is("hydra"));
        assert_eq!(format!("{a}"), "chimera");
    }
}
