const TableBody = ({ tableData, columns }) => {
    return (
      <tbody>
        {tableData.map((data) => {
          return (
            <tr key={data.id}>
              {columns.map(({ accessor }) => {
                const tData = data[accessor] ? data[accessor] : "——";
                return <td key={accessor}>{tData}</td>;
              })}
              <button className="action-button buy-button">Buy</button>
              <button className="action-button sell-button">Sell</button>
            </tr>
          );
        })}
      </tbody>
    );
  };
  
  export default TableBody;