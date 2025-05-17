#    Dynamic-QNetSim: a simulator for comparative evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

class ForwardingInformationBase:
    def __init__(self):
        # The FIB table stores multiple path entries
        self.table: dict[str, dict] = {}

    def add_entry(self, path_id: str, path_vector: list[str], swap_sequence: list[int],
                  purification_scheme: dict[tuple[str, str], int], qubit_addresses: list[int]):
        """Add a new path entry to the forwarding table."""
        if path_id in self.table:
            raise ValueError(f"Path ID '{path_id}' already exists.")

        self.table[path_id] = {
            "path_id": path_id,
            "path_vector": path_vector,
            "swap_sequence": swap_sequence,
            "purification_scheme": purification_scheme,
            "qubit_addresses": qubit_addresses
        }

    def get_entry(self, path_id: str) -> dict | None:
        """Retrieve an entry from the table."""
        return self.table.get(path_id, None)

    def update_entry(self, path_id: str, **kwargs):
        """Update an existing entry with new data."""
        if path_id not in self.table:
            raise KeyError(f"Path ID '{path_id}' not found.")

        for key, value in kwargs.items():
            if key == "path_id":
                continue
            if key in self.table[path_id]:
                self.table[path_id][key] = value
            else:
                raise KeyError(f"Invalid key '{key}' for update.")

    def delete_entry(self, path_id: str):
        """Remove an entry from the table."""
        if path_id in self.table:
            del self.table[path_id]
        else:
            raise KeyError(f"Path ID '{path_id}' not found.")

    def __repr__(self):
        """Return a string representation of the forwarding table."""
        return "\n".join(
            f"Path ID: {path_id}, Path: {entry['path_vector']}, Swap Sequence: {entry['swap_sequence']}, "
            f"Purification: {entry['purification_scheme']}, Qubit Addresses: {entry['qubit_addresses']}"
            for path_id, entry in self.table.items()
        )
