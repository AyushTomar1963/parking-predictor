/**
 * LotsList component
 * Dropdown selector for parking lots
 */
export default function LotsList({ lots, selected, onChange }) {
  return (
    <div className="form-group">
      <label htmlFor="lot-select">Parking Lot:</label>
      <select
        id="lot-select"
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        className="form-select"
      >
        {lots.length === 0 && (
          <option value="">Loading lots...</option>
        )}
        {lots.map((lot) => {
          const lotId = typeof lot === 'string' ? lot : (lot.lot_id || lot.file || lot.name);
          return (
            <option key={lotId} value={lotId}>
              {lotId}
            </option>
          );
        })}
      </select>
    </div>
  )
}
